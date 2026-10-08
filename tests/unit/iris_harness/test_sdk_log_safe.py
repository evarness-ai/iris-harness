from iris_harness.foundation import logsafe
from iris_harness.sdk import logging as sdk_logging


def test_the_sdk_name_is_the_core_function_itself() -> None:
    assert sdk_logging.log_safe is logsafe.log_safe
