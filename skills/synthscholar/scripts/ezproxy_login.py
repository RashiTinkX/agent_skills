"""Sign in to an institutional EZproxy in a browser and remember the session.

Users should not have to know what an EZproxy host, a rewrite mode or a
cookies.txt export is. This module replaces the three ``EZPROXY_*`` variables
with one browser sign-in:

1. **Which proxy.** Worked out, in order, from: an explicit ``--proxy-host``;
   a link the user copied from their library (``--library-link``); an
   institution name or email domain (``--institution``); the saved login; the
   ``EZPROXY_HOST`` variable; the domain of the user's email
   (``SYNTHSCHOLAR_EMAIL`` / ``NCBI_EMAIL`` / ``git config user.email``). A
   domain is mapped through ``KNOWN_PROXIES``, then by probing the usual
   EZproxy hostnames for that domain — a candidate is used only if the server
   answers as EZproxy.
2. **Sign-in.** A real browser window opens at the proxy's login page; the user
   signs in with their library account (SSO, MFA — whatever their institution
   uses; nothing here sees the password). The window closes by itself once the
   proxy lets them through.
3. **Remembered.** The proxy host, the rewrite mode (detected from where the
   login landed) and the proxy's own session cookies are saved under
   ``~/.config/synthscholar/`` (owner-only permissions). Later runs reuse them,
   and the browser only opens again when the session has expired.

Only cookies for the proxy's own domain are saved — not the identity
provider's — so the file grants exactly what a logged-in proxy tab would, until
the proxy expires the session. ``--logout`` deletes all of it.

Needs ``playwright`` for the browser step (``pip install playwright``). It drives
the installed Chrome or Edge when there is one; otherwise it needs
``playwright install chromium`` once.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlparse

import httpx

MODE_HOSTNAME_SUFFIX = "hostname-suffix"
MODE_LOGIN_URL = "login-url"

# A page every EZproxy licenses: the login lands here, which proves the session
# and shows the rewrite convention in the resulting URL.
PROBE_TARGET = "https://www.nature.com/"
LOGIN_TIMEOUT = 300  # seconds the user has to finish signing in

# Email / institution domain → EZproxy host, for institutions confirmed to use
# this form. Anything missing is found by probing (below) or from a library link.
KNOWN_PROXIES = {
    "mit.edu": "libproxy.mit.edu",
}
# Hostnames institutions commonly give their EZproxy, tried in order.
PROBE_PATTERNS = ("ezproxy.{d}", "libproxy.{d}", "ezproxy.lib.{d}", "proxy.lib.{d}",
                  "login.ezproxy.{d}", "{stem}.idm.oclc.org")

# Exit codes the CLI uses so a caller (the agent) knows what to ask the user.
EXIT_NEEDS_INSTITUTION = 4
EXIT_NEEDS_BROWSER = 3


class LoginError(RuntimeError):
    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code


# ─────────────────────────── saved state ────────────────────────────

def state_dir() -> Path:
    base = os.environ.get("SYNTHSCHOLAR_CONFIG_DIR") or (
        Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "synthscholar")
    d = Path(base).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    try:
        d.chmod(0o700)
    except OSError:
        pass
    return d


def _state_file() -> Path:
    return state_dir() / "ezproxy.json"


def cookie_path() -> Path:
    return state_dir() / "ezproxy-cookies.txt"


def load_saved() -> Optional[dict]:
    """The saved login, or None. Ignored when its cookie file is gone."""
    try:
        data = json.loads(_state_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not data.get("host") or not Path(data.get("cookie_file", "")).is_file():
        return None
    return data


def _write_private(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(path, 0o600)


def save(host: str, mode: str, cookies: list[dict], source: str) -> dict:
    lines = ["# Netscape HTTP Cookie File — EZproxy session saved by synthscholar"]
    for c in cookies:
        domain = c.get("domain", "")
        lines.append("\t".join([
            domain, "TRUE" if domain.startswith(".") else "FALSE", c.get("path") or "/",
            "TRUE" if c.get("secure") else "FALSE",
            str(int(c.get("expires") or 0)) if (c.get("expires") or 0) > 0 else "0",
            c["name"], c.get("value", "")]))
    _write_private(cookie_path(), "\n".join(lines) + "\n")
    state = {"host": host, "mode": mode, "cookie_file": str(cookie_path()),
             "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "host_source": source}
    _write_private(_state_file(), json.dumps(state, indent=2) + "\n")
    return state


def logout() -> list[str]:
    """Delete the saved login, cookies and browser profile; return what was removed."""
    removed = []
    for p in (_state_file(), cookie_path()):
        if p.exists():
            p.unlink()
            removed.append(str(p))
    profile = state_dir() / "browser-profile"
    if profile.exists():
        shutil.rmtree(profile, ignore_errors=True)
        removed.append(str(profile))
    return removed


# ─────────────────────────── which proxy ────────────────────────────

def host_from_link(link: str) -> Optional[str]:
    """Proxy host from any link the user's library gave them.

    ``https://www-nature-com.libproxy.mit.edu/articles/…`` → ``libproxy.mit.edu``;
    ``https://ezproxy.uni.edu/login?url=…`` → ``ezproxy.uni.edu``;
    ``https://login.ezproxy.uni.edu/login?qurl=…`` → ``ezproxy.uni.edu``.
    """
    p = urlparse(link.strip())
    netloc = (p.netloc or "").lower().split("@")[-1].split(":")[0]
    if not netloc:
        return None
    if p.path.lower().startswith("/login") or re.search(r"[?&]q?url=", p.query or ""):
        return netloc[len("login."):] if netloc.startswith("login.") else netloc
    first, _, rest = netloc.partition(".")
    # a rewritten publisher host: "www-nature-com" — dashes where dots were
    if "-" in first and rest.count(".") >= 1:
        return rest
    return netloc


def _email_domains() -> list[str]:
    emails = [os.environ.get("SYNTHSCHOLAR_EMAIL", ""), os.environ.get("NCBI_EMAIL", "")]
    if shutil.which("git"):
        try:
            emails.append(subprocess.run(["git", "config", "user.email"], capture_output=True,
                                         text=True, timeout=5).stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
    return [e.split("@", 1)[1].lower() for e in emails if "@" in e]


def _registrable(domain: str) -> str:
    """``eecs.mit.edu`` → ``mit.edu``; ``cam.ac.uk`` stays (two-part public suffix)."""
    parts = domain.lower().strip(".").split(".")
    if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in ("ac", "edu", "co", "org", "gov"):
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def is_ezproxy(host: str, timeout: float = 8.0) -> bool:
    """Does ``host`` answer as an EZproxy server?"""
    try:
        r = httpx.get(f"https://{host}/login", timeout=timeout, follow_redirects=False)
    except httpx.HTTPError:
        return False
    server = (r.headers.get("server") or "").lower()
    loc = (r.headers.get("location") or "").lower()
    return "ezproxy" in server or "ezproxy" in r.text[:4000].lower() or "ezproxy" in loc


def host_for_domain(domain: str) -> Optional[str]:
    d = _registrable(domain)
    if d in KNOWN_PROXIES:
        return KNOWN_PROXIES[d]
    stem = d.split(".")[0]
    for pat in PROBE_PATTERNS:
        cand = pat.format(d=d, stem=stem)
        if is_ezproxy(cand):
            return cand[len("login."):] if cand.startswith("login.") else cand
    return None


def resolve_host(proxy_host: str = "", library_link: str = "", institution: str = "") -> tuple[str, str]:
    """(host, how it was found). Raises LoginError(EXIT_NEEDS_INSTITUTION)."""
    if proxy_host:
        return proxy_host.strip().lower(), "given"
    if library_link:
        h = host_from_link(library_link)
        if h:
            return h, "library link"
    if institution:
        dom = institution.strip().lower()
        dom = dom.split("@", 1)[1] if "@" in dom else dom
        if "." not in dom:  # a bare name like "mit": try it as <name>.edu
            dom = f"{dom}.edu"
        h = host_for_domain(dom)
        if h:
            return h, f"institution {institution}"
        # asked for a specific institution: never fall back to someone else's proxy
        raise LoginError(
            f"Couldn't find a library proxy for '{institution}'. Ask the user for a link "
            "they get when opening an article through their library, and pass it as "
            "--library-link.", EXIT_NEEDS_INSTITUTION)
    if library_link:
        raise LoginError(f"'{library_link}' doesn't look like a library proxy link.",
                         EXIT_NEEDS_INSTITUTION)
    saved = load_saved()
    if saved:
        return saved["host"], "saved login"
    if os.environ.get("EZPROXY_HOST"):
        return os.environ["EZPROXY_HOST"].strip().lower(), "EZPROXY_HOST"
    for dom in _email_domains():
        h = host_for_domain(dom)
        if h:
            return h, f"email domain {dom}"
    raise LoginError(
        "Couldn't tell which library proxy to use. Ask the user for their university or "
        "library (e.g. 'MIT'), or a link they get when opening an article through their "
        "library, and pass it as --institution or --library-link.", EXIT_NEEDS_INSTITUTION)


# ─────────────────────────── session ────────────────────────────

def login_url(host: str, target: str = PROBE_TARGET) -> str:
    """The proxy's starting URL for ``target``.

    ``qurl=`` carries the target percent-encoded; ``url=`` must be raw, and an
    encoded value there doesn't match any configured database, so EZproxy shows
    its menu page instead of the article.
    """
    return f"https://{host}/login?qurl={quote(target, safe='')}"


def _cookie_dict(cookie_file: str) -> dict[str, str]:
    out = {}
    try:
        for line in Path(cookie_file).read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 7 and not line.startswith("#"):
                out[parts[5]] = parts[6]
    except OSError:
        pass
    return out


def session_valid(host: str, cookie_file: str, timeout: float = 20.0) -> Optional[bool]:
    """True = the proxy lets this session through; False = expired; None = can't tell."""
    jar = httpx.Cookies()
    for name, value in _cookie_dict(cookie_file).items():
        jar.set(name, value, domain=f".{host}")
    try:
        r = httpx.get(login_url(host), cookies=jar,
                      timeout=timeout, follow_redirects=True)
    except httpx.HTTPError:
        return None
    final = urlparse(str(r.url))
    netloc = final.netloc.lower()
    through = netloc.endswith("." + host) or (netloc == host and not final.path.lower().startswith("/login"))
    return bool(through and r.status_code < 400)


def _proxy_cookies(cookies: list[dict], host: str) -> list[dict]:
    return [c for c in cookies
            if (c.get("domain") or "").lstrip(".").endswith(host)]


def browser_login(host: str, source: str, timeout: int = LOGIN_TIMEOUT) -> dict:
    """Open a browser at the proxy's login, wait for the user, save the session."""
    try:
        from playwright.sync_api import Error as PwError
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise LoginError("Signing in needs Playwright: ask the user, then "
                         "`pip install playwright` (and `playwright install chromium` "
                         "if they have neither Chrome nor Edge).", EXIT_NEEDS_BROWSER)

    profile = state_dir() / "browser-profile"   # keeps the IdP's "remember me"
    start = login_url(host)
    print(f"Opening a browser window — sign in with your library account ({host}).\n"
          f"It closes by itself once you're through (waiting up to {timeout // 60} min).")
    with sync_playwright() as p:
        ctx = None
        errors = []
        for channel in ("chrome", "msedge", None):
            try:
                ctx = p.chromium.launch_persistent_context(
                    str(profile), headless=False, channel=channel, chromium_sandbox=True,
                    viewport={"width": 1100, "height": 800})
                break
            except PwError as e:
                errors.append(f"{channel or 'chromium'}: {str(e).splitlines()[0]}")
        if ctx is None:
            raise LoginError("No browser Playwright can drive (" + "; ".join(errors) + "). "
                             "Ask the user, then run `playwright install chromium`.",
                             EXIT_NEEDS_BROWSER)
        try:
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            try:
                page.goto(start, wait_until="domcontentloaded", timeout=30000)
            except PwError as e:
                raise LoginError(f"Couldn't open {start}: {str(e).splitlines()[0]}. Is {host} "
                                 "your library's proxy? Pass --library-link or --institution.")
            deadline = time.monotonic() + timeout
            mode = None
            while time.monotonic() < deadline:
                if not ctx.pages:
                    raise LoginError("The browser window was closed before sign-in finished.")
                for pg in ctx.pages:
                    u = urlparse(pg.url)
                    netloc = u.netloc.lower()
                    on_login = u.path.lower().startswith(("/login", "/connect"))
                    if netloc.endswith("." + host) and not netloc.startswith("login.") \
                            or netloc == "login." + host and not on_login:
                        # a rewritten publisher host, or the proxy's menu after sign-in
                        mode = MODE_HOSTNAME_SUFFIX
                    elif netloc == host and not on_login \
                            and any(c["name"].lower().startswith("ezproxy")
                                    for c in _proxy_cookies(ctx.cookies(), host)):
                        mode = MODE_LOGIN_URL
                if mode:
                    break
                # not time.sleep: the sync API only processes browser events
                # inside its own calls, so pg.url would never change
                ctx.pages[0].wait_for_timeout(1000)
            if not mode:
                raise LoginError(f"Sign-in didn't finish within {timeout // 60} minutes.")
            cookies = _proxy_cookies(ctx.cookies(), host)
        finally:
            try:
                ctx.close()
            except Exception:
                pass
    if not cookies:
        raise LoginError("Signed in, but the proxy set no session cookie to save.")
    return save(host, mode, cookies, source)


def ensure_session(*, proxy_host: str = "", library_link: str = "", institution: str = "",
                   force: bool = False, timeout: int = LOGIN_TIMEOUT) -> dict:
    """A working saved session: reuse it if still valid, otherwise sign in."""
    saved = load_saved()
    host, source = resolve_host(proxy_host, library_link, institution)
    if saved and saved["host"] == host and not force:
        ok = session_valid(host, saved["cookie_file"])
        if ok is not False:   # valid, or unknown (offline) — don't reopen a browser blindly
            return saved
        print(f"Your {host} session has expired — signing in again.")
    return browser_login(host, source, timeout=timeout)
