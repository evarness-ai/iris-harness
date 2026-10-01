"""Is a result written in the reader's language? A writing-script check, nothing heavier.

The research engine's ``language`` option (the digest's ``news_language``) asks the
provider for that language where it can, then drops any result whose title is mostly
in another script: a Japanese headline in an English digest. It cannot tell two
languages of one script apart (English from German): the provider hint does that part.
No model and no word lists; the only table is ISO 639-1 code -> the Unicode scripts
the language is written in. A code not in it is not filtered (the hint still goes out).
For a non-Latin language, Latin letters (brand names, "AI") do not count against it.
"""

from __future__ import annotations

import unicodedata

# ISO 639-1 code -> the leading word(s) of ``unicodedata.name`` for its letters.
_SCRIPTS: dict[str, tuple[str, ...]] = {
    **dict.fromkeys(
        ("en", "de", "es", "fr", "it", "nl", "pt", "sv", "da", "no", "fi", "pl"), ("LATIN",)
    ),
    "ru": ("CYRILLIC",),
    "uk": ("CYRILLIC",),
    "el": ("GREEK",),
    "ar": ("ARABIC",),
    "he": ("HEBREW",),
    "hi": ("DEVANAGARI",),
    "mr": ("DEVANAGARI",),
    "bn": ("BENGALI",),
    "ta": ("TAMIL",),
    "te": ("TELUGU",),
    "kn": ("KANNADA",),
    "ml": ("MALAYALAM",),
    "ja": ("CJK", "HIRAGANA", "KATAKANA"),
    "zh": ("CJK",),
    "ko": ("HANGUL",),
}

_LATIN = ("LATIN",)

# Share of a title's letters that must be in the language's script.
MIN_SCRIPT_SHARE = 0.8


def in_language(text: str, language: str | None) -> bool:
    """``text`` is mostly written in ``language``'s script.

    True when there is nothing to judge: no language, a language this check does not
    know, or text with no letters (a title of digits and symbols).
    """
    if not language or language not in _SCRIPTS:
        return True
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return True
    scripts = _SCRIPTS[language]
    if scripts != _LATIN:
        # Latin letters are neutral in a non-Latin language's headline ("生成AI"),
        # but a headline with no other letters at all is not in that language.
        letters = [ch for ch in letters if not _script(ch).startswith(_LATIN)]
        if not letters:
            return False
    hits = sum(1 for ch in letters if _script(ch).startswith(scripts))
    return hits / len(letters) >= MIN_SCRIPT_SHARE


def _script(ch: str) -> str:
    return unicodedata.name(ch, "")


__all__ = ["MIN_SCRIPT_SHARE", "in_language"]
