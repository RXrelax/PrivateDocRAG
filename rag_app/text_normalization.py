import re
import unicodedata


INVISIBLE_CHARS_PATTERN = re.compile(r"[\u200b-\u200f\ufeff]")

# PDF extraction can emit CJK radical glyphs instead of ordinary Han characters.
# NFKC fixes compatibility ideographs such as ⽩/⾻, but not these radicals.
CJK_RADICAL_TRANSLATION = str.maketrans(
    {
        "⻣": "骨",
        "⻓": "长",
        "⻅": "见",
        "⻢": "马",
        "⻉": "贝",
        "⻤": "鬼",
        "⻜": "飞",
        "⻰": "龙",
        "⻩": "黄",
        "⻛": "风",
        "⻔": "门",
        "⻋": "车",
        "⻥": "鱼",
        "⻦": "鸟",
        "⻮": "齿",
        "⻘": "青",
        "⻝": "食",
        "⻄": "西",
    }
)


def normalize_cjk_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.translate(CJK_RADICAL_TRANSLATION)
    return INVISIBLE_CHARS_PATTERN.sub("", normalized)


def normalize_searchable_text(text: str) -> str:
    normalized = normalize_cjk_text(text).lower()
    return re.sub(r"\s+", " ", normalized).strip()
