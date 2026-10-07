import logging

import pytest

from iris_harness.foundation.observability.logging_setup import log_ingress


def test_a_decoded_newline_in_the_path_cannot_forge_a_second_log_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        log_ingress(method="GET", path="/x\nINGRESS POST /admin", session_id="s\r\n1", status=200)
    lines = [r.getMessage() for r in caplog.records if "INGRESS" in r.getMessage()]
    assert len(lines) == 1
    assert "\n" not in lines[0] and "\r" not in lines[0]
