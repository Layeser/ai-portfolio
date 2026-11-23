"""Text normalisation shared by BM25, chunk matching and answer checking."""

import re
import unicodedata

FRENCH_STOPWORDS = set(
    """a au aux avec ce ces cet cette d dans de des du elle en est et etre il ils je l la le les leur lui
    ma mais me mes mon ne nos notre nous on ou par pas pour qu que qui sa se ses son sur ta te tes ton tu
    un une vos votre vous y quel quelle quels quelles combien comment dois doit puis peut faut ai j m
    s si sont ont avoir fait mon ma mes moi""".split()
)


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalize(text: str) -> str:
    """Lowercase, no accents, '1 500' -> '1500', '2,50' -> '2.50', single spaces."""
    text = strip_accents(text.lower()).replace("’", "'")
    text = re.sub(r"(?<=\d)[\s  ](?=\d{3}\b)", "", text)
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """BM25 tokens: normalised words minus stopwords, with a light plural stemming."""
    tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", normalize(text))
    return [t[:-1] if len(t) > 4 and t.endswith("s") else t for t in tokens if t not in FRENCH_STOPWORDS]


def contains_fact(text: str, fact: str) -> bool:
    """Numbers match by value (25 != 250, 2.50 == 2.5); other facts match as normalised substrings."""
    text, fact = normalize(text), normalize(fact)
    if re.fullmatch(r"\d+(?:\.\d+)?", fact):  # compare numbers by value: "2.50" == "2.5", "250" != "25"
        return any(float(n) == float(fact) for n in re.findall(r"\d+(?:\.\d+)?", text))
    return fact in text
