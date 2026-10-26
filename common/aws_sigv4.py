"""SigV4 request signing for httpx.

Deliberately generic: this signs any outbound HTTP request for any AWS service.
It knows nothing about AgentCore, and it is the only AWS-specific code in the
request path. Swap it out and the agents work against a plain HTTP endpoint
unchanged -- which is the point of keeping the auth concern in one small,
injectable object rather than threading a platform SDK through the agents.

It is built on botocore, which boto3 already brings in, so this adds no
dependency.

Why not a bearer token: with IAM there is no client secret to store, no token
endpoint to reach, and no expiry to manage. The caller's identity is its
execution role, which is also what the gateway's Cedar policies match on as
AgentCore::IamEntity. One identity mechanism instead of two.
"""
from __future__ import annotations

import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
from botocore.session import Session as BotocoreSession

# Headers botocore must not sign: hop-by-hop headers and anything httpx
# recalculates after signing. Signing these produces a mismatch and a 403 that
# gives no hint as to why.
_UNSIGNABLE = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "expect",
}


class SigV4HttpxAuth(httpx.Auth):
    """Sign each request with SigV4 for a given service and region.

    requires_request_body is True because SigV4 hashes the payload: the body
    has to be available before the signature can be computed.
    """

    requires_request_body = True

    def __init__(
        self,
        service: str,
        region: str,
        credentials: Credentials | None = None,
        botocore_session: BotocoreSession | None = None,
    ) -> None:
        self._service = service
        self._region = region
        self._static_credentials = credentials
        self._session = botocore_session or BotocoreSession()

    def _credentials(self) -> Credentials:
        if self._static_credentials is not None:
            return self._static_credentials

        resolved = self._session.get_credentials()
        if resolved is None:
            raise RuntimeError(
                "No AWS credentials found for SigV4 signing. On AgentCore the "
                "runtime execution role supplies them; locally export a profile "
                "(AWS_PROFILE) or run 'aws login'."
            )
        # Resolve per call: on a runtime these are refreshable and rotate.
        return resolved.get_frozen_credentials()

    def auth_flow(self, request: httpx.Request):
        body = request.content or b""

        signable_headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in _UNSIGNABLE
        }

        aws_request = AWSRequest(
            method=request.method,
            url=str(request.url),
            data=body,
            headers=signable_headers,
        )
        SigV4Auth(self._credentials(), self._service, self._region).add_auth(
            aws_request
        )

        # Copy the signature material back. Content-Length is excluded because
        # httpx owns it and a stale value breaks the request.
        for key, value in aws_request.headers.items():
            if key.lower() == "content-length":
                continue
            request.headers[key] = value

        yield request


def signer_for_agentcore(region: str) -> SigV4HttpxAuth:
    """SigV4 signer for AgentCore Runtime and Gateway endpoints.

    'bedrock-agentcore' is the SigV4 service name for both.
    """
    return SigV4HttpxAuth(service="bedrock-agentcore", region=region)
