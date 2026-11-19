"""Keyword search over the chunks with BM25, as a counterpart to the vector search in app.search.

The index is built in memory from the chunks table. keyword_search() returns the same shape as
search_chunks(), but SearchHit.similarity holds the BM25 score: higher is better, and it is not
a 0-1 cosine similarity, so the two kinds of score can't be compared directly.
"""

import re
from functools import lru_cache

import psycopg
from rank_bm25 import BM25Okapi

from app.db import connect
from app.search import SearchHit

# A plain English stopword list; contractions are listed without apostrophes because the
# tokenizer drops them ("don't" -> "dont").
STOPWORDS = frozenset("""
a about above after again against all am an and any are arent as at be because been before being
below between both but by can cannot cant could couldnt did didnt do does doesnt doing dont down
during each few for from further had hadnt has hasnt have havent having he hed hell hes her here
heres hers herself him himself his how hows i id ill im ive if in into is isnt it its itself lets
me more most mustnt my myself no nor not of off on once only or other ought our ours ourselves out
over own same shant she shed shell shes should shouldnt so some such than that thats the their
theirs them themselves then there theres these they theyd theyll theyre theyve this those through
to too under until up very was wasnt we wed well were weve werent what whats when whens where
wheres which while who whos whom why whys will with wont would wouldnt you youd youll youre youve
your yours yourself yourselves also may might must shall upon yet thus
""".split())

APOSTROPHES = re.compile(r"['’‘]")
PUNCTUATION = re.compile(r"[^\w\s]|_")


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, drop English stopwords."""
    text = PUNCTUATION.sub(" ", APOSTROPHES.sub("", text.lower()))
    return [word for word in text.split() if word not in STOPWORDS]


class KeywordIndex:
    def __init__(self, rows: list[tuple[str, str, str, str, str]]) -> None:
        """rows: (chunk_id, author, work, part, text) for every chunk."""
        self.rows = rows
        self.bm25 = BM25Okapi([tokenize(r[4]) for r in rows])

    @classmethod
    def from_db(cls, conn: psycopg.Connection | None = None) -> "KeywordIndex":
        query = "SELECT chunk_id, author, work, part, text FROM chunks ORDER BY chunk_id"
        if conn is not None:
            return cls(conn.execute(query).fetchall())
        with connect() as conn:
            return cls(conn.execute(query).fetchall())

    def search(self, question: str, k: int) -> list[SearchHit]:
        """The k best-scoring chunks, best first; chunks sharing no keyword are left out."""
        scores = self.bm25.get_scores(tokenize(question))
        ranked = sorted(range(len(self.rows)), key=lambda i: scores[i], reverse=True)
        return [
            SearchHit(chunk_id=r[0], author=r[1], work=r[2], part=r[3],
                      similarity=float(scores[i]), text=r[4])
            for i in ranked[:k]
            if scores[i] > 0
            for r in [self.rows[i]]
        ]


@lru_cache(maxsize=1)
def get_keyword_index() -> KeywordIndex:
    """Built on first use and kept; rebuild with get_keyword_index.cache_clear() after a reload."""
    return KeywordIndex.from_db()


def keyword_search(question: str, k: int) -> list[SearchHit]:
    """BM25 counterpart of search_chunks(): the k chunks that best match the question's keywords."""
    return get_keyword_index().search(question, k)
