"""Build the direct-code-deployment zip for AgentCore Runtime.

All six runtimes share ONE zip. Each runtime differs only by its entryPoint and
environment variables, so there is nothing to gain from six packages and a lot
to lose in drift between them.

Direct code deployment, rather than containers, because it removes the whole
Docker/ECR step: zip the code and its dependencies, put it in S3, point the
runtime at it. Documented limits are 250 MB zipped and 750 MB unzipped; this
package lands around 50 MB zipped.

The one real constraint: AgentCore Runtime is arm64 Linux, so the dependencies
must be Linux aarch64 wheels, not the macOS wheels in your local .venv. We
resolve them with uv for the target platform and force wheel-only installs --
if a package has no aarch64 wheel, failing loudly here is much better than
discovering it in a runtime import error.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = ROOT / ".build"
VENDOR_DIR = BUILD_DIR / "package"
ZIP_PATH = BUILD_DIR / "travel_agent_deploy.zip"

TARGET_PLATFORM = "aarch64-manylinux2014"
TARGET_PYTHON = "3.12"

# Project code that goes in the zip. The knowledge-base markdown and the
# fixtures travel with it -- they are the demo's data layer, not external state.
INCLUDE_PATHS = [
    "agents",
    "common",
    "mcp_servers",
    "knowledgebase",
    "fixtures",
]

EXCLUDE_NAMES = {
    "__pycache__", ".venv", ".build", ".run", ".git", ".DS_Store",
    ".pytest_cache", ".ruff_cache",
}

# Documented AgentCore limits for a .zip deployment package.
MAX_ZIPPED_MB = 250
MAX_UNZIPPED_MB = 750


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-4000:] + "\n" + result.stderr[-4000:] + "\n")
        raise RuntimeError(f"command failed: {' '.join(cmd[:4])} ...")


def _dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def vendor_dependencies() -> None:
    """Install linux/aarch64 wheels into the package directory."""
    print(f"  resolving dependencies for {TARGET_PLATFORM} (python {TARGET_PYTHON})")
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError(
            "uv is required to cross-resolve arm64 wheels from macOS. "
            "Install it with: brew install uv"
        )
    _run([
        uv, "pip", "install",
        "--python-platform", TARGET_PLATFORM,
        "--python-version", TARGET_PYTHON,
        "--target", str(VENDOR_DIR),
        # Wheel-only: a source distribution would be built for the HOST
        # architecture and silently ship a macOS binary to an arm64 Linux box.
        "--only-binary=:all:",
        "--no-cache",
        "-r", str(ROOT / "requirements.txt"),
    ])


def copy_project_code() -> None:
    print("  copying project code")
    for rel in INCLUDE_PATHS:
        source = ROOT / rel
        if not source.exists():
            raise RuntimeError(f"expected {source} to exist")
        shutil.copytree(
            source,
            VENDOR_DIR / rel,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(*EXCLUDE_NAMES, "*.pyc", ".kb_index.json"),
        )


def normalise_permissions() -> None:
    """AgentCore needs 644 on files and 755 on directories.

    A zip built from a Mac can carry group/other-unreadable modes, and the
    runtime then cannot read its own code. Cheap to fix, obscure to debug.
    """
    print("  normalising POSIX permissions (644 files / 755 dirs)")
    for path in VENDOR_DIR.rglob("*"):
        if path.is_dir():
            os.chmod(path, 0o755)
        elif path.is_file():
            os.chmod(path, 0o755 if path.suffix in {".sh", ".so"} else 0o644)
    os.chmod(VENDOR_DIR, 0o755)


def build_zip() -> Path:
    print(f"  writing {ZIP_PATH.name}")
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    written = 0
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(VENDOR_DIR.rglob("*")):
            # Exclusions must be checked against the path RELATIVE to the
            # package root. Checking path.parts would match ".build" on every
            # single entry, because VENDOR_DIR itself lives under .build --
            # which silently produced an empty zip and an AgentCore
            # CREATE_FAILED with "entrypoint could not be found".
            relative = path.relative_to(VENDOR_DIR)
            if any(part in EXCLUDE_NAMES for part in relative.parts):
                continue

            arcname = str(relative)
            info = zipfile.ZipInfo(arcname + ("/" if path.is_dir() else ""))
            # Preserve the modes we just set; ZipFile.write would not.
            info.external_attr = (0o755 if path.is_dir() else
                                  (path.stat().st_mode & 0o777)) << 16
            if path.is_dir():
                info.external_attr |= 0x10
                zf.writestr(info, b"")
            else:
                info.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(info, path.read_bytes())
                written += 1

    if written == 0:
        raise RuntimeError(
            f"packaged 0 files into {ZIP_PATH.name}. Check EXCLUDE_NAMES "
            f"against the paths under {VENDOR_DIR}."
        )
    print(f"  packaged {written} files")
    return ZIP_PATH


def build(clean: bool = True) -> Path:
    """Build the deployment package and return the zip path."""
    print("packaging for AgentCore Runtime (direct code deployment)")
    if clean and VENDOR_DIR.exists():
        shutil.rmtree(VENDOR_DIR)
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)

    vendor_dependencies()
    copy_project_code()
    normalise_permissions()
    zip_path = build_zip()

    unzipped = _dir_size_mb(VENDOR_DIR)
    zipped = zip_path.stat().st_size / (1024 * 1024)
    print(f"  package: {zipped:.1f} MB zipped, {unzipped:.1f} MB unzipped")

    if zipped > MAX_ZIPPED_MB:
        raise RuntimeError(
            f"zip is {zipped:.0f} MB, over the {MAX_ZIPPED_MB} MB limit."
        )
    if unzipped > MAX_UNZIPPED_MB:
        raise RuntimeError(
            f"unzipped is {unzipped:.0f} MB, over the {MAX_UNZIPPED_MB} MB limit."
        )

    # Fail here rather than at runtime if a macOS binary sneaked in.
    verify_arm64(zip_path)
    verify_entrypoints(zip_path)
    return zip_path


# Every entryPoint deploy.py registers with AgentCore. If one of these is
# missing from the zip, the runtime fails with CREATE_FAILED and
# "The specified entrypoint could not be found or accessed in your artifact."
REQUIRED_ENTRYPOINTS = [
    "mcp_servers/traveller_profile/server.py",
    "mcp_servers/journey_search/server.py",
    "mcp_servers/ticket_operations/server.py",
    "agents/journey_planner/server.py",
    "agents/journey_support/server.py",
    "agents/policy_advisor/server.py",
    "agents/train_travel_agent/server.py",
]

# Imports the entrypoints make at module load, plus the data files they read.
# A missing dependency here is a runtime ImportError, which surfaces as an
# opaque container failure. Note langgraph is a PEP 420 namespace package with
# no top-level __init__.py, so these name real modules rather than package
# roots.
REQUIRED_SUPPORT = [
    "common/config.py",
    "common/observability.py",
    "common/aws_sigv4.py",
    "common/policy_shim.py",
    # The router imports these at startup: a2a_discovery fetches every
    # sub-agent's card to build its delegation tools, and config_store resolves
    # the registry that tells it which agents to look for. Missing either one
    # is an ImportError in the first second of the container's life, which
    # AgentCore reports as "Runtime initialization time exceeded".
    "common/a2a_discovery.py",
    "common/config_store.py",
    "common/kb_tool.py",
    "common/memory.py",
    "fixtures/travellers.json",
    "fixtures/services.json",
    "fixtures/fares.json",
    "fixtures/bookings.json",
    "fixtures/stations.json",
    "knowledgebase/fare-rules.md",
    "knowledgebase/indexer.py",
    # third-party
    "mcp/server/fastmcp/__init__.py",
    "langgraph/prebuilt/__init__.py",
    "langgraph/graph/state.py",
    "langchain_aws/__init__.py",
    "langchain_mcp_adapters/client.py",
    "a2a/server/apps/__init__.py",
    "a2a/client/__init__.py",
    "botocore/auth.py",
    "httpx/__init__.py",
    "uvicorn/__init__.py",
    "starlette/applications.py",
    # tracing
    "opentelemetry/sdk/trace/__init__.py",
    "opentelemetry/instrumentation/httpx/__init__.py",
    "opentelemetry/instrumentation/starlette/__init__.py",
    "opentelemetry/sdk/extension/aws/trace/__init__.py",
    "opentelemetry/propagators/aws/__init__.py",
    "opentelemetry/exporter/otlp/proto/http/trace_exporter/__init__.py",
]


def verify_entrypoints(zip_path: Path) -> None:
    """Assert the zip actually contains what the runtimes will be told to run.

    This is the check that was missing when an over-broad exclusion produced an
    empty zip: AgentCore read the artifact fine and reported the entrypoint
    missing, which is a long way from the real cause.
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())

    missing_entry = [p for p in REQUIRED_ENTRYPOINTS if p not in names]
    missing_support = [p for p in REQUIRED_SUPPORT if p not in names]
    if missing_entry or missing_support:
        raise RuntimeError(
            "the package is missing files the runtimes need:\n"
            + "".join(f"  ENTRYPOINT MISSING: {p}\n" for p in missing_entry)
            + "".join(f"  SUPPORT MISSING:    {p}\n" for p in missing_support)
            + f"  (zip contains {len(names)} entries)"
        )
    print(
        f"  verified all {len(REQUIRED_ENTRYPOINTS)} entrypoints and "
        f"{len(REQUIRED_SUPPORT)} support paths are present"
    )


def verify_arm64(zip_path: Path) -> None:
    """Assert the native extensions in the zip are Linux aarch64."""
    import re

    bad = []
    checked = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".so"):
                continue
            # ELF e_machine at offset 18; 0xB7 == AArch64.
            with zf.open(name) as fh:
                header = fh.read(20)
            if len(header) < 20 or header[:4] != b"\x7fELF":
                bad.append(f"{name} (not an ELF binary -- wrong platform)")
                continue
            if int.from_bytes(header[18:20], "little") != 0xB7:
                bad.append(f"{name} (ELF but not AArch64)")
            checked += 1
            if re.search(r"macosx|darwin", name):
                bad.append(f"{name} (macOS artefact)")

    if bad:
        raise RuntimeError(
            "the package contains non-arm64 native binaries:\n  "
            + "\n  ".join(bad[:10])
        )
    if checked == 0:
        raise RuntimeError(
            "found no native extensions in the package at all. The dependency "
            "vendoring step produced nothing -- pydantic-core and orjson alone "
            "should contribute several .so files."
        )
    print(f"  verified {checked} native extension(s) are Linux aarch64")


if __name__ == "__main__":
    build()
