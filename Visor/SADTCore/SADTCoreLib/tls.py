"""HTTPS only, verified against the public certificate authorities.

Standard library only, `certifi` aside (imported when first needed), so `deploy.py` (urllib) and `client.py` (requests) share
it without either one pulling in the other's stack.

Two rules live here and nowhere else, and neither has an off switch -- no
setting, flag, environment variable or config value relaxes them:

1. **The server URL is `https://`, always.** This client sends medical images
   and a bearer token, and the server has no plain-HTTP listener at all. Any
   `http://` URL is refused, whatever the host, loopback included: nothing is
   upgraded behind the user's back, and an `http://` URL saved by an older
   client is refused with the same message.

2. **The certificate is verified against the public authorities**, the
   `certifi` bundle, named explicitly (`ca_bundle()`) for `requests` and for
   urllib alike, with the hostname check on. There is no custom CA, no
   pinning and never `verify=False` -- and not `verify=True` either, which
   would let `REQUESTS_CA_BUNDLE`/`CURL_CA_BUNDLE` in the environment swap the
   trusted authorities for any others. The server is therefore reached through the name its
   publicly-signed certificate is issued for -- the
   `https://<name>.<tailnet>.ts.net` address `tailscale serve` publishes, or
   its own domain behind a reverse proxy or served directly with a Let's
   Encrypt or institutional certificate -- including when it runs on this
   very machine: the self-signed certificate the server presents on its own
   port is not one this client accepts.
"""

import ipaddress
import ssl
import urllib.error
import urllib.request
from urllib.parse import urlsplit

HTTPS_REQUIRED_MESSAGE = (
    "The server URL must start with https://; this server only speaks HTTPS "
    "and medical images are never sent unencrypted."
)


class InsecureServerUrl(ValueError):
    """The server URL is not https://."""

    def __init__(self, message: str = HTTPS_REQUIRED_MESSAGE):
        super().__init__(message)


def is_loopback(host) -> bool:
    """Whether `host` names this machine, by name or by address.

    The client's one loopback predicate. `127.0.0.1` is not the only loopback
    address: the whole `127.0.0.0/8` is, and so is `::1`, which arrives from a
    URL wrapped in brackets, and any `*.localhost` name. It decides nothing
    about transport security -- loopback is held to the same https:// rule --
    only whether the data leaves the machine (see settings_qt).
    """
    if not host:
        return False
    host = host.strip("[]").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def require_https(url: str) -> str:
    """Return `url` without a trailing slash, or raise InsecureServerUrl.

    Anything but `https://<host>` is refused, `http://localhost` included.
    """
    text = (url or "").strip().rstrip("/")
    try:
        parts = urlsplit(text)
        host = parts.hostname
        parts.port  # noqa: B018 - raises ValueError on a malformed port
    except ValueError:
        raise InsecureServerUrl() from None
    if parts.scheme.lower() != "https" or not host:
        raise InsecureServerUrl()
    return "https" + text[len(parts.scheme):]


def ca_bundle() -> str:
    """The public certificate authorities every connection is verified
    against: certifi's bundle, the one `requests` ships. Passed explicitly,
    never left to `verify=True`, so no environment variable can replace it.
    Imported here rather than at the top so the module stays importable where
    only the standard library is."""
    import certifi

    return certifi.where()


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Every redirect is an error. A redirect is how https:// becomes http://
    without the new URL ever passing require_https."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(
            req.full_url, code, "redirects are never followed", headers, fp
        )


class _RefusePlainHttp(urllib.request.HTTPHandler):
    def http_open(self, req):
        raise urllib.error.URLError(HTTPS_REQUIRED_MESSAGE)


def _opener(context: ssl.SSLContext):
    return urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=context), _RefuseRedirects(), _RefusePlainHttp()
    )


def urlopen(url: str, timeout: float):
    """urllib's urlopen, https:// only, never following a redirect, verified
    against the public certificate authorities with the hostname check on."""
    url = require_https(url)
    return _opener(ssl.create_default_context(cafile=ca_bundle())).open(url, timeout=timeout)
