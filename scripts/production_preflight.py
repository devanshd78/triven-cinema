#!/usr/bin/env python3
import os
import shutil
import socket
import stat
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
ERRORS: list[str] = []
WARNINGS: list[str] = []


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    WARNINGS.append(message)
    print(f"[WARN] {message}")


def fail(message: str) -> None:
    ERRORS.append(message)
    print(f"[FAIL] {message}")


def read_env() -> dict[str, str]:
    values: dict[str, str] = {}
    if not ENV_PATH.exists():
        return values
    for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def command_ok(*command: str) -> bool:
    try:
        return subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        ).returncode == 0
    except Exception:
        return False


def port_in_use(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.3)
    try:
        return sock.connect_ex(("127.0.0.1", port)) == 0
    finally:
        sock.close()


def hostname_resolves(hostname: str) -> bool:
    try:
        socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
        return True
    except socket.gaierror:
        return False


def main() -> int:
    print("Triven Cinema Hostinger VPS production preflight")
    print(f"Project: {ROOT}")

    if sys.platform.startswith("linux"):
        ok("Linux host detected")
    else:
        warn(f"Current host is {sys.platform!r}; production target is a Hostinger Linux VPS")

    if not ENV_PATH.exists():
        fail(".env is missing. Copy .env.production.example to .env and add secrets.")
        env = {}
    else:
        env = read_env()
        mode = stat.S_IMODE(ENV_PATH.stat().st_mode)
        if mode & 0o077:
            warn(f".env permissions are {oct(mode)}; run: chmod 600 .env")
        else:
            ok(".env permissions are private")

    required = [
        "TRIVEN_DOMAIN",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "MODAL_APP_NAME",
        "MODAL_FUNCTION_NAME",
    ]
    for key in required:
        if env.get(key):
            ok(f"{key} configured")
        else:
            fail(f"{key} is missing")

    domain = env.get("TRIVEN_DOMAIN", "").strip()
    if domain and domain != "devansh.info":
        fail("This demo deployment is pinned to TRIVEN_DOMAIN=devansh.info")
    if domain:
        parsed = urlparse(domain if "://" in domain else f"https://{domain}")
        if "://" in domain:
            fail("TRIVEN_DOMAIN must be a hostname only, for example devansh.info (no https://)")
        elif not parsed.hostname or parsed.hostname != domain:
            fail("TRIVEN_DOMAIN is not a valid hostname")
        elif hostname_resolves(domain):
            ok(f"DNS resolves for {domain}")
        else:
            warn(f"DNS does not currently resolve for {domain}; Nginx/Certbot HTTPS cannot issue a public certificate yet")

        frontend_url = env.get("FRONTEND_URL", "").rstrip("/")
        expected_url = f"https://{domain}"
        if frontend_url == expected_url:
            ok("FRONTEND_URL matches TRIVEN_DOMAIN")
        else:
            warn(f"FRONTEND_URL should normally be {expected_url!r}")

    if env.get("APP_ENV", "").lower() in {"production", "prod"}:
        ok("APP_ENV=production")
    else:
        fail("APP_ENV must be production")

    if env.get("DEBUG", "").lower() in {"false", "0", "no"}:
        ok("DEBUG=false")
    else:
        fail("DEBUG must be false")

    if not env.get("CORS_ORIGINS", "").strip():
        ok("CORS_ORIGINS is empty for same-origin production traffic")
    else:
        warn("CORS_ORIGINS is set; same-origin Hostinger deployment normally does not need CORS")

    if env.get("AUTH_ENABLED", "true").lower() in {"true", "1", "yes"}:
        if len(env.get("TRIVEN_SECRET_KEY", "")) >= 32:
            ok("TRIVEN_SECRET_KEY configured for signed login sessions")
        else:
            fail("TRIVEN_SECRET_KEY must contain at least 32 characters for production login")
        if env.get("DEMO_AUTH_SHOW_OTP", "true").lower() in {"true", "1", "yes"}:
            fail("DEMO_AUTH_SHOW_OTP must be false in production; configure SMTP email delivery")
        for key in ("SMTP_HOST", "SMTP_FROM_EMAIL"):
            if not env.get(key):
                fail(f"{key} is required for production login")
        if env.get("SMTP_USE_TLS", "true").lower() not in {"true", "1", "yes"} and env.get("SMTP_USE_SSL", "false").lower() not in {"true", "1", "yes"}:
            fail("SMTP must use TLS or SSL in production")
    else:
        fail("AUTH_ENABLED must be true in production")

    if env.get("VIDEO_PROVIDER") == "modal":
        ok("VIDEO_PROVIDER=modal")
    else:
        warn("VIDEO_PROVIDER is not modal")

    if env.get("ENABLE_SYNC_RENDER_ENDPOINTS", "true").lower() in {"false", "0", "no"}:
        ok("Synchronous paid render endpoints disabled")
    else:
        warn("ENABLE_SYNC_RENDER_ENDPOINTS should be false in production")

    if shutil.which("docker"):
        ok("docker available")
        if command_ok("docker", "compose", "version"):
            ok("docker compose available")
        else:
            fail("docker compose is unavailable")
    else:
        fail("docker is not installed")

    compose_file = ROOT / "docker-compose.production.yml"
    if compose_file.exists() and shutil.which("docker"):
        if command_ok("docker", "compose", "-f", str(compose_file), "config", "-q"):
            ok("docker-compose.production.yml validates")
        else:
            fail("docker-compose.production.yml failed validation; check .env values")

    storage = ROOT / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    if os.access(storage, os.W_OK):
        ok("storage directory is writable")
    else:
        fail("storage directory is not writable")

    free_gb = shutil.disk_usage(storage).free / (1024**3)
    try:
        minimum = float(env.get("MINIMUM_FREE_DISK_GB", "10") or 10)
    except ValueError:
        minimum = 10.0
    if free_gb >= minimum:
        ok(f"Disk free: {free_gb:.1f} GiB")
    else:
        fail(f"Only {free_gb:.1f} GiB free; configured minimum is {minimum:.1f} GiB")

    if env.get("GEMINI_API_KEY"):
        ok("GEMINI_API_KEY configured")
    else:
        warn("GEMINI_API_KEY is empty; multi-scene planning will use the local fallback")

    billing_enabled = env.get("BILLING_ENABLED", "false").lower() in {"true", "1", "yes"}
    if billing_enabled:
        for key in ("TRIVEN_SECRET_KEY", "STRIPE_SECRET_KEY", "STRIPE_WEBHOOK_SECRET"):
            if env.get(key):
                ok(f"{key} configured for billing")
            else:
                fail(f"{key} is required when BILLING_ENABLED=true")
        if any(env.get(key) for key in ("STRIPE_PRICE_STARTER", "STRIPE_PRICE_PRO", "STRIPE_PRICE_STUDIO")):
            ok("At least one Stripe Checkout price is configured")
        else:
            fail("Configure at least one Stripe price when BILLING_ENABLED=true")
    else:
        ok("Billing integration is feature-gated off")

    youtube_enabled = env.get("YOUTUBE_ENABLED", "false").lower() in {"true", "1", "yes"}
    if youtube_enabled:
        for key in ("TRIVEN_SECRET_KEY", "YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET"):
            if env.get(key):
                ok(f"{key} configured for YouTube")
            else:
                fail(f"{key} is required when YOUTUBE_ENABLED=true")
        callback = env.get("YOUTUBE_REDIRECT_URI", "").strip() or (
            f"https://{domain}/api/v1/youtube/callback" if domain else ""
        )
        if callback:
            ok(f"YouTube OAuth callback: {callback}")
    else:
        ok("YouTube integration is feature-gated off")

    try:
        native_chunk = float(env.get("LTX_NATIVE_CHUNK_SECONDS", "10") or 10)
        max_1080 = float(env.get("MAX_1080P_SCENE_SECONDS", "30") or 30)
        max_4k = float(env.get("MAX_4K_SCENE_SECONDS", "15") or 15)
        if native_chunk <= 0 or max_1080 < native_chunk or max_4k <= 0:
            fail("Long-form duration profile is invalid")
        else:
            ok(f"Duration profiles configured: native chunk {native_chunk:g}s, 1080p {max_1080:g}s, 4K {max_4k:g}s")
    except ValueError:
        fail("Duration profile values must be numeric")

    if env.get("TRIVEN_LTX_REPO_REF", "main") == "main":
        warn("TRIVEN_LTX_REPO_REF=main is not reproducible; pin a validated commit/tag")
    else:
        ok("LTX repository revision is pinned")

    if (ROOT / ".git").exists():
        tracked = subprocess.run(
            ["git", "ls-files", ".env"], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()
        if tracked:
            fail(".env is tracked by git")
        else:
            ok(".env is not tracked by git")
        tracked_state = subprocess.run(
            ["git", "ls-files", "storage"], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()
        if tracked_state:
            fail("Runtime storage is tracked by Git. Back it up and remove it from the index before deployment.")

    nginx_template = ROOT / "deploy" / "hostinger" / "nginx.triven-cinema.conf"
    if nginx_template.exists():
        ok("Host Nginx reverse-proxy template is present")
    else:
        fail("deploy/hostinger/nginx.triven-cinema.conf is missing")

    for port in (80, 443):
        if port_in_use(port):
            if shutil.which("nginx"):
                ok(f"Port {port} is in use; host Nginx is expected to own public HTTP/HTTPS")
            else:
                warn(f"Port {port} is already in use; verify the owning reverse proxy")
        else:
            warn(f"Port {port} is not listening yet; enable Nginx/Certbot before public launch")

    if shutil.which("ufw"):
        ok("UFW command available")
    else:
        warn("ufw is not installed; configure Hostinger firewall or an equivalent host firewall")

    print(f"\nSummary: {len(ERRORS)} error(s), {len(WARNINGS)} warning(s)")
    return 1 if ERRORS else 0


if __name__ == "__main__":
    raise SystemExit(main())
