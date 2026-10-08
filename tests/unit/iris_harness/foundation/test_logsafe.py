from iris_harness.foundation.logsafe import log_safe


def test_control_characters_are_escaped_so_one_value_is_one_line() -> None:
    assert log_safe("a\r\nINFO forged\x1b[0m") == "a\\r\\nINFO forged\\x1b[0m"


def test_long_values_are_truncated() -> None:
    out = log_safe("x" * 500)
    assert out == "x" * 200 + "..."
