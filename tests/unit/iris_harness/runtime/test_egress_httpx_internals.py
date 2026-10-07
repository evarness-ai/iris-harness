"""PinnedTransport swaps a private httpx attribute; a library upgrade that moves it must fail here."""

import httpcore
import httpx


def test_httpx_transport_still_keeps_its_pool_in_the_private_attribute():
    transport = httpx.HTTPTransport()
    try:
        assert isinstance(transport._pool, httpcore.ConnectionPool)
    finally:
        transport.close()
