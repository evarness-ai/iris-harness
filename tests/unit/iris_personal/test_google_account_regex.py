import pytest

from iris_personal.connections import google


def test_valid_address_is_normalised() -> None:
    assert google._normalize_account(" Owner@Example.COM ") == "owner@example.com"


@pytest.mark.parametrize("bad", ["a@b..c", "a@b.c.", "a@.c", "a@b", "a b@c.d", "@c.d"])
def test_malformed_addresses_are_refused(bad: str) -> None:
    with pytest.raises(google.StartRefused):
        google._normalize_account(bad)


def test_length_cap_and_pathological_input_are_cheap() -> None:
    ok = "a" * 240 + "@b.co"  # 245
    assert google._normalize_account(ok) == ok
    with pytest.raises(google.StartRefused):
        google._normalize_account("a" * 250 + "@b.co")  # 255
    with pytest.raises(google.StartRefused):
        google._normalize_account("a@" + ".!" * 100000)
