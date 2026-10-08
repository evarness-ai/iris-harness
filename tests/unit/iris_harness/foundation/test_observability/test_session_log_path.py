import pytest

from iris_harness.foundation.observability import session_log
from iris_harness.foundation.observability.trace_builder import get_trace, session_messages


@pytest.fixture(autouse=True)
def _logs(tmp_path, monkeypatch):
    monkeypatch.setattr(session_log, "LOG_DIR", tmp_path)
    return tmp_path


def test_ordinary_ids_resolve_inside_the_log_dir(tmp_path) -> None:
    assert session_log.session_log_path("telegram:42") == tmp_path / "session-telegram:42.jsonl"


@pytest.mark.parametrize("bad", ["../escape", "a/../../b", "x/y", "\x00z"])
def test_ids_that_leave_the_log_dir_are_refused(bad: str) -> None:
    with pytest.raises(ValueError):
        session_log.session_log_path(bad)


def test_readers_treat_an_escaping_id_as_unknown() -> None:
    assert session_messages("../../etc/passwd") == []
    assert get_trace("../../etc/passwd~0") is None
