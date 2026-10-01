"""The research language check: a headline's writing script, no model, no word lists."""

from __future__ import annotations

import pytest

from iris_harness.plugins_builtin.research.language import in_language


@pytest.mark.parametrize(
    "title",
    [
        "OpenAI ships a new model",
        "Café owners in São Paulo react",  # accented Latin is still Latin
        "Nvidia's Q3: $35.1B revenue, +94%",
    ],
)
def test_latin_script_titles_are_english(title: str) -> None:
    assert in_language(title, "en")


@pytest.mark.parametrize(
    "title",
    [
        "生成AIの新モデル、国内企業が相次ぎ導入",  # the owner's real complaint: Japanese
        "中国人工智能新闻",
        "Новости искусственного интеллекта",
        "कृत्रिम बुद्धिमत्ता समाचार",
    ],
)
def test_other_scripts_are_not_english(title: str) -> None:
    assert not in_language(title, "en")


def test_a_few_foreign_letters_do_not_sink_an_english_title() -> None:
    assert in_language("Sony launches the PlayStation 6 in Tokyo (東京)", "en")


def test_the_check_works_for_other_languages_too() -> None:
    assert in_language("生成AIの新モデル", "ja")
    assert not in_language("OpenAI ships a new model", "ja")


def test_nothing_to_judge_passes() -> None:
    assert in_language("2026 — 1:0", "en")  # no letters
    assert in_language("生成AI", None)  # no language asked for
    assert in_language("生成AI", "xx")  # a code the check does not know
