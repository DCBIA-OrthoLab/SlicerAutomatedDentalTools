"""A throwaway certificate authority for the tests that run a real server.

The client speaks HTTPS only and verifies every certificate against the public
authorities, with no knob to change that. So a test server on 127.0.0.1 gets a
certificate for `localhost` issued by a throwaway CA made here, and the TEST --
never the production code -- patches `tls.ca_bundle` to name that CA for its
own duration. The client still verifies and checks the hostname, exactly as in
production; only the list of trusted authorities differs.
"""

import atexit
import os
import shutil
import ssl
import subprocess
import tempfile
import unittest
from unittest import mock

_DIRECTORY = None
# Why no certificate could be made, once that is known: every later test is
# skipped with the same reason instead of trying again.
_UNAVAILABLE = None


_SKIP_REASON = ("no way to build a TLS test certificate on this machine: neither the "
                "cryptography package nor a working openssl")


def _usable(directory):
    """Would a client trusting ca.pem complete a handshake with cert.pem for
    `localhost`, in THIS Python's ssl module? Done in memory, with no socket:
    a certificate that was written but that this OpenSSL refuses (an SHA-1
    signature from macOS's LibreSSL is refused as CA_MD_TOO_WEAK) is no more
    use than none."""
    path = lambda name: os.path.join(directory, name)  # noqa: E731
    try:
        server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_context.load_cert_chain(path("cert.pem"), path("key.pem"))
        client_context = ssl.create_default_context(cafile=path("ca.pem"))
        to_server, to_client = ssl.MemoryBIO(), ssl.MemoryBIO()
        server_in, server_out = ssl.MemoryBIO(), ssl.MemoryBIO()
        client = client_context.wrap_bio(to_client, to_server, server_hostname="localhost")
        server = server_context.wrap_bio(server_in, server_out, server_side=True)
        done = {client: False, server: False}
        for _ in range(10):
            for end in (client, server):
                if not done[end]:
                    try:
                        end.do_handshake()
                        done[end] = True
                    except ssl.SSLWantReadError:
                        pass
            server_in.write(to_server.read())
            to_client.write(server_out.read())
            if all(done.values()):
                return True
        return False
    except (OSError, ssl.SSLError, ValueError):
        return False


def _generate(directory):
    """`ca.pem`, `cert.pem` and `key.pem` in `directory`, or raise
    unittest.SkipTest. The `cryptography` package first, the `openssl`
    program second, and each result is kept only if it actually works here.
    The extension factory's machines may have neither: no `openssl` on the
    PATH (Linux), one failing on these options (Windows), LibreSSL signing
    with SHA-1 (macOS), and a Slicer Python without `cryptography`. The tests
    needing a TLS server are then reported as skipped, not failed: what they
    would test is the client, not the machine's tooling."""
    for generate in (_generate_with_cryptography, _generate_with_openssl):
        try:
            generate(directory)
        except (ImportError, OSError, subprocess.SubprocessError, ValueError):
            continue
        if _usable(directory):
            return
    raise unittest.SkipTest(_SKIP_REASON)


def _generate_with_cryptography(directory):
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    now = datetime.datetime.now(datetime.timezone.utc)
    valid_from, valid_until = now - datetime.timedelta(hours=1), now + datetime.timedelta(days=2)

    def name(common_name):
        return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])

    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = name("throwaway test CA")
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name).issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(valid_from).not_valid_after(valid_until)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(x509.KeyUsage(
            digital_signature=False, content_commitment=False, key_encipherment=False,
            data_encipherment=False, key_agreement=False, key_cert_sign=True, crl_sign=True,
            encipher_only=False, decipher_only=False), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )

    key = ec.generate_private_key(ec.SECP256R1())
    cert = (
        x509.CertificateBuilder()
        .subject_name(name("localhost")).issuer_name(ca_name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(valid_from).not_valid_after(valid_until)
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
                       critical=False)
        .sign(ca_key, hashes.SHA256())
    )

    def write(file_name, data):
        with open(os.path.join(directory, file_name), "wb") as handle:
            handle.write(data)

    write("ca.pem", ca_cert.public_bytes(serialization.Encoding.PEM))
    write("cert.pem", cert.public_bytes(serialization.Encoding.PEM))
    write("key.pem", key.private_bytes(serialization.Encoding.PEM,
                                       serialization.PrivateFormat.PKCS8,
                                       serialization.NoEncryption()))


def _openssl(*args):
    subprocess.run(["openssl", *args], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _generate_with_openssl(directory):
    path = lambda name: os.path.join(directory, name)  # noqa: E731
    ec = ["-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes"]
    _openssl("req", "-x509", *ec, "-keyout", path("ca.key"), "-out", path("ca.pem"),
             "-days", "2", "-subj", "/CN=throwaway test CA",
             "-addext", "basicConstraints=critical,CA:TRUE",
             "-addext", "keyUsage=critical,keyCertSign,cRLSign")
    _openssl("req", *ec, "-keyout", path("key.pem"), "-out", path("server.csr"),
             "-subj", "/CN=localhost")
    with open(path("ext.cnf"), "w") as handle:
        handle.write("subjectAltName=DNS:localhost\nbasicConstraints=critical,CA:FALSE\n"
                     "extendedKeyUsage=serverAuth\n")
    _openssl("x509", "-req", "-in", path("server.csr"), "-CA", path("ca.pem"),
             "-CAkey", path("ca.key"), "-CAcreateserial", "-out", path("cert.pem"),
             "-days", "2", "-extfile", path("ext.cnf"))


def _directory():
    """The certificates' directory, or unittest.SkipTest -- raised from the
    test's setUp or body, which unittest reports as a skip."""
    global _DIRECTORY, _UNAVAILABLE
    if _UNAVAILABLE is not None:
        raise unittest.SkipTest(_UNAVAILABLE)
    if _DIRECTORY is None:
        directory = tempfile.mkdtemp(prefix="tls_fixture_")
        atexit.register(shutil.rmtree, directory, True)
        try:
            _generate(directory)
        except unittest.SkipTest as skip:
            _UNAVAILABLE = str(skip)
            raise
        _DIRECTORY = directory
    return _DIRECTORY


def ca_bundle():
    return os.path.join(_directory(), "ca.pem")


def serve_tls(server):
    """Wrap a socketserver's listening socket in TLS, in place, with the
    `localhost` certificate. The handshake is deferred to the first read, in
    the handler thread, so one client failing it never blocks serve_forever.
    """
    try:
        directory = _directory()
    except unittest.SkipTest:
        server.server_close()  # bound already by the caller; do not leak it
        raise
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(os.path.join(directory, "cert.pem"), os.path.join(directory, "key.pem"))
    server.socket = context.wrap_socket(server.socket, server_side=True, do_handshake_on_connect=False)
    return server


def trust_test_ca(testcase):
    """For this test only, make the client trust the throwaway CA, by
    replacing the one function that names the trusted authorities."""
    from ServerToolsCoreLib import tls

    patcher = mock.patch.object(tls, "ca_bundle", return_value=ca_bundle())
    patcher.start()
    testcase.addCleanup(patcher.stop)
