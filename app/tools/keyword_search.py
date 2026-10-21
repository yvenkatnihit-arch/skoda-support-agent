"""BM25 keyword search over the chunk texts: the keyword half of hybrid search.

Vector search finds passages with the same meaning; BM25 finds passages that contain the exact words
(ISOFIX, TPMS, 1.5 TSI, Android Auto), which embeddings can rank too low. Built in memory from
chunks.json when the vector store loads, so it needs no extra files and no API calls.

Queries are expanded with a small synonym list (boot -> luggage, mileage -> kmpl ...) because customers and
brochures often use different words for the same thing. Expansion terms count half.
"""

import math
import re
import unicodedata
from collections import Counter

BM25_K1 = 1.5  # how quickly repeated words stop adding score
BM25_B = 0.75  # how much long chunks are penalised
SYNONYM_WEIGHT = 0.5

STOPWORDS = frozenset("""
a an and are as at be but by can do does for from has have how i in is it its me my of on or our so
than that the their them there these this to was what when where which who why will with you your
skoda car cars please tell about want know
""".split())

# Each group: words customers use interchangeably. A query word from a group adds the other members.
SYNONYM_GROUPS = [
    {"boot", "luggage", "cargo", "trunk", "dickey"},
    {"mileage", "kmpl", "efficiency", "economy"},
    {"adas", "driver assistance"},
    {"rsa", "roadside", "breakdown"},
    {"infotainment", "touchscreen", "screen", "display"},
    {"ev", "electric", "battery"},
    {"seat", "seating", "seater"},
    {"ncap", "crash", "rating"},
    {"warranty", "guarantee"},
    {"sunroof", "moonroof", "panoramic"},
    {"bhp", "ps", "power", "horsepower"},
    {"dsg", "automatic"},
]


def _fold(text: str) -> str:
    """Lower case without accents, so 'Škoda' matches 'skoda'."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _stem(word: str) -> str:
    """Very light plural folding: airbags -> airbag, litres -> litre (enough for brochure vocabulary)."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")  # keeps decimals like 1.5 together


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in TOKEN_RE.findall(_fold(text)) if t not in STOPWORDS]


def _synonym_index() -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for group in SYNONYM_GROUPS:
        tokens = {t for phrase in group for t in tokenize(phrase)}
        for t in tokens:
            index.setdefault(t, set()).update(tokens - {t})
    return index


SYNONYMS = _synonym_index()


def expand_query(query: str) -> dict[str, float]:
    """Query terms with weights: 1.0 for the customer's own words, SYNONYM_WEIGHT for added synonyms."""
    weights: dict[str, float] = {}
    for term in tokenize(query):
        weights[term] = 1.0
    for term in list(weights):
        for extra in SYNONYMS.get(term, ()):
            weights.setdefault(extra, SYNONYM_WEIGHT)
    return weights


class BM25Index:
    """Okapi BM25 over a list of texts; document i is chunk id i."""

    def __init__(self, texts: list[str]):
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.docs else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {term: math.log((n - f + 0.5) / (f + 0.5) + 1) for term, f in df.items()}

    def score(self, doc_id: int, terms: dict[str, float]) -> float:
        doc, length = self.docs[doc_id], self.lengths[doc_id]
        total = 0.0
        for term, weight in terms.items():
            tf = doc.get(term)
            if not tf:
                continue
            norm = BM25_K1 * (1 - BM25_B + BM25_B * length / (self.avg_length or 1))
            total += weight * self.idf[term] * tf * (BM25_K1 + 1) / (tf + norm)
        return total

    def search(self, query: str, allowed_ids: list[int], k: int) -> list[tuple[int, float]]:
        """Top-k (chunk id, BM25 score) among allowed_ids; chunks sharing no query word are left out."""
        terms = expand_query(query)
        if not terms:
            return []
        scored = [(i, self.score(i, terms)) for i in allowed_ids]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]
