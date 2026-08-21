"""
retrieval.py — the "brain" of the tax Q&A chatbot.

Given a user's question, it decides HOW to answer using three strategies,
tried in order. The first one that produces an answer wins:

  1. Exact tax lookup      (SQLite)  -> tax brackets, standard deductions
  2. Exact historical lookup (SQLite) -> IRS filing counts by year
  3. Document search       (PDFs)    -> everything else, via hybrid search

Strategies 1 and 2 give precise, computed answers from structured tables.
Strategy 3 is the fallback: search the IRS PDFs semantically.
"""

import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from embedding import OpenAIEmbeddingsAdapter

# ---------------------------------------------------------------------------
# CONFIG — where things live on disk, and tuning knobs in one place
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "DATA"          # source PDFs
DB_PATH = BASE_DIR / "db" / "sqlite.db"      # structured tax tables
CHROMA_PATH = BASE_DIR / "db" / "chroma"     # vector database

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How much each signal counts in the final ranking (see hybrid_search).
RERANK_WEIGHT = 1.0
DENSE_WEIGHT = 0.1
BM25_WEIGHT = 0.05

_reranker = None  # loaded once, on first use (it's a heavy model)


def tokenize(text):
    """Split text into lowercase words. 'Hello, World!' -> ['hello', 'world']"""
    return re.findall(r"\w+", text.lower())


# ===========================================================================
# PART 1 — VECTOR STORE ACCESS
# Each PDF has its own Chroma collection of embedded text chunks.
# ===========================================================================
def get_store(collection_name, persist_directory=CHROMA_PATH):
    """Open one PDF's vector collection."""
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_directory),
        embedding_function=OpenAIEmbeddingsAdapter(),
    )


def get_stores(persist_directory=CHROMA_PATH):
    """Open a vector collection for every PDF in the DATA folder."""
    return [
        get_store(pdf.stem, persist_directory=persist_directory)
        for pdf in sorted(DATA_DIR.glob("*.pdf"))
    ]


# ===========================================================================
# PART 2 — THREE WAYS TO SCORE A CHUNK AGAINST THE QUERY
# ===========================================================================
def dense_search(store, query, k=50):
    """
    VECTOR search: find chunks whose *meaning* is close to the query.
    Returns the top-k chunks, each with a distance score (lower = closer).
    """
    hits = store.similarity_search_with_score(query, k=k)
    return [
        {
            "text": doc.page_content,
            "dense_score": float(distance),
            "metadata": doc.metadata or {},
        }
        for doc, distance in hits
    ]


def bm25_scores(query, texts):
    """
    KEYWORD search (BM25): reward chunks that share exact words with the
    query, giving rare words more weight than common ones. This catches
    things vector search can miss, like specific IDs or number strings.
    Returns one score per text (higher = better keyword match).
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return [0.0] * len(texts)

    docs = [tokenize(text) for text in texts]
    # How many chunks contain each word (used to down-weight common words).
    doc_freq = Counter({word for tokens in docs for word in set(tokens)})
    avg_len = sum(len(tokens) for tokens in docs) / max(len(docs), 1)
    k1, b = 1.5, 0.75  # standard BM25 tuning constants

    scores = []
    for tokens in docs:
        counts = Counter(tokens)
        score = 0.0
        for word in query_tokens:
            if word not in counts:
                continue
            idf = max(0.0, (len(docs) - doc_freq[word] + 0.5) / (doc_freq[word] + 0.5))
            freq = counts[word]
            denom = freq + k1 * (1 - b + b * len(tokens) / avg_len)
            score += idf * freq * (k1 + 1) / denom
        scores.append(score)
    return scores


def _get_reranker():
    """Load the cross-encoder once and reuse it (it's slow to initialise)."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(query, texts):
    """
    RERANKING: a small model reads (query, chunk) pairs together and scores
    how well each chunk actually answers the query. Slower but far more
    accurate than vector distance, so we only run it on the shortlist.
    """
    if not texts:
        return []
    pairs = [(query, text) for text in texts]
    scores = _get_reranker().predict(pairs, batch_size=16, show_progress_bar=False)
    return [float(score) for score in scores]


# ===========================================================================
# PART 3 — HYBRID SEARCH: combine the three signals
# ===========================================================================
def hybrid_search(store, query, top_k=10, dense_k=50):
    """
    1. Cast a wide net with fast vector search (dense_k candidates).
    2. Score that shortlist three ways: rerank, vector, keyword.
    3. Blend the scores and return the best top_k.
    """
    candidates = dense_search(store, query, k=dense_k)
    if not candidates:
        return []

    texts = [c["text"] for c in candidates]
    bm25 = bm25_scores(query, texts)
    reranked = rerank(query, texts)

    results = []
    for hit, bm25_score, rerank_score in zip(candidates, bm25, reranked):
        # Convert vector distance (lower=better) into a similarity (higher=better).
        dense_similarity = 1 / (1 + hit["dense_score"])
        final_score = (
            RERANK_WEIGHT * rerank_score
            + DENSE_WEIGHT * dense_similarity
            + BM25_WEIGHT * bm25_score
        )
        results.append({
            "text": hit["text"],
            "metadata": hit["metadata"],
            "dense_score": hit["dense_score"],
            "bm25_score": bm25_score,
            "rerank_score": rerank_score,
            "hybrid_score": final_score,
        })

    results.sort(key=lambda r: r["hybrid_score"], reverse=True)
    return results[:top_k]


# ===========================================================================
# PART 4 — UNDERSTAND THE QUESTION (pull structured fields out of free text)
# ===========================================================================
def _filing_status(query):
    """Detect the tax filing status mentioned in the query, if any."""
    text = query.lower().replace("-", " ").replace("_", " ")
    # Longer phrases are checked first so "married filing jointly" wins over "jointly".
    statuses = {
        "married filing jointly": "married_filing_jointly",
        "married jointly": "married_filing_jointly",
        "jointly": "married_filing_jointly",
        "married filing separately": "married_filing_separately",
        "married separately": "married_filing_separately",
        "separately": "married_filing_separately",
        "head of household": "head_of_household",
        "single": "single",
    }
    for phrase, status in sorted(statuses.items(), key=lambda kv: -len(kv[0])):
        if phrase in text:
            return status
    return None


def _income(query):
    """Extract a dollar income figure, understanding '$80,000' and '80k'."""
    patterns = (
        r"(?:make|earn|income|salary|wage)[^$\d]{0,20}\$?([\d,]+(?:\.\d+)?)\s*(k|thousand)?",
        r"\$([\d,]+(?:\.\d+)?)\s*(k|thousand)?",
    )
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            value = float(match.group(1).replace(",", ""))
            if match.group(2):          # "k" or "thousand" was present
                value *= 1000
            return value
    return None


# ===========================================================================
# PART 5 — SMALL DATABASE HELPERS
# ===========================================================================
def _table_with_columns(connection, required_columns):
    """Find the first table that contains all of the given column names."""
    tables = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    for (table_name,) in tables:
        safe_name = table_name.replace('"', '""')
        columns = {row[1] for row in connection.execute(f'PRAGMA table_info("{safe_name}")')}
        if required_columns.issubset(columns):
            return table_name
    return None


def _source_file(table_name):
    """Map a table name back to the PDF/file it came from, for citations."""
    normalized_table = re.sub(r"[^a-z0-9]", "", table_name.lower())
    for path in DATA_DIR.iterdir():
        normalized_stem = re.sub(r"[^a-z0-9]", "", path.stem.lower())
        if (
            normalized_stem == normalized_table
            or normalized_table.startswith(normalized_stem)
            or normalized_stem.startswith(normalized_table)
        ):
            return path.name
    return table_name


def _pdf_document(store):
    """Best-guess filename of the PDF behind a vector store, for citations."""
    name = getattr(store, "_collection_name", "unknown")
    candidate = DATA_DIR / f"{name}.pdf"
    return candidate.name if candidate.exists() else f"{name}.pdf"


def _source_filename(value):
    """Return only the filename portion of a source metadata value."""
    return Path(str(value)).name


# ===========================================================================
# PART 6 — EXACT ANSWERS FROM SQLITE
# ===========================================================================
def sqlite_tax_answer(query, db_path=DB_PATH):
    """
    Answer tax-bracket / standard-deduction questions with exact numbers.
    Only fires when the query clearly asks for tax data AND names a filing
    status AND gives an income (or asks for the deduction). Otherwise
    returns None so the caller falls through to the next strategy.
    """
    q = query.lower()
    asks_tax_data = any(
        term in q
        for term in ("tax bracket", "marginal rate", "standard deduction", "taxable income")
    )
    income = _income(query)
    status = _filing_status(query)
    asks_deduction = "standard deduction" in q

    if not asks_tax_data or status is None or (income is None and not asks_deduction):
        return None
    if not db_path.exists():
        return None

    year_match = re.search(r"\b(20\d{2})\b", query)
    tax_year = int(year_match.group(1)) if year_match else 2026

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        table = _table_with_columns(
            connection,
            {"tax_year", "filing_status", "marginal_rate_percent", "standard_deduction_usd"},
        )
        if table is None:
            return None

        if income is None:
            # Just the standard deduction for this status/year.
            row = connection.execute(
                f'SELECT * FROM "{table}" WHERE tax_year = ? AND filing_status = ? LIMIT 1',
                (tax_year, status),
            ).fetchone()
        else:
            # Find the bracket the income falls into.
            row = connection.execute(
                f"""
                SELECT * FROM "{table}"
                WHERE tax_year = ? AND filing_status = ?
                  AND (taxable_income_over_usd IS NULL OR taxable_income_over_usd <= ?)
                  AND (taxable_income_up_to_usd IS NULL OR ? < taxable_income_up_to_usd)
                ORDER BY taxable_income_over_usd IS NOT NULL DESC,
                         taxable_income_over_usd DESC
                LIMIT 1
                """,
                (tax_year, status, income, income),
            ).fetchone()

    if row is None:
        return None

    status_text = status.replace("_", " ")
    if income is None:
        answer = (
            f"For {tax_year} {status_text}, the standard deduction is "
            f"${row['standard_deduction_usd']:,.0f}."
        )
    else:
        upper = row["taxable_income_up_to_usd"]
        upper_text = f"up to ${upper:,.0f}" if upper is not None else "above the last threshold"
        answer = (
            f"For {tax_year} {status_text}, taxable income of ${income:,.0f} falls in the "
            f"{row['marginal_rate_percent']}% marginal tax bracket ({upper_text}). "
            f"The standard deduction is ${row['standard_deduction_usd']:,.0f}."
        )

    return {
        "answer": answer,
        "source": "SQLite",
        "source_filename": _source_file(table),
        "metadata": {"source_url": row["primary_source_url"], "tax_year": tax_year, "table": table},
    }


def sqlite_historical_answer(query, db_path=DB_PATH):
    """
    Answer historical questions like "how many returns were filed in 2015".
    The historical table is laid out with years across the top and metrics
    down the side, so we locate the right column (year) and row (metric).
    Returns None if the query isn't clearly historical or nothing matches.
    """
    year_match = re.search(r"\b(199\d|20(?:0\d|1\d|2[0-3]))\b", query)
    if year_match is None:
        return None

    q = query.lower()
    history_terms = ("historical", "history", "all returns", "electronically filed", "income tax returns")
    if not any(term in q for term in history_terms):
        return None
    if not db_path.exists():
        return None

    tax_year = int(year_match.group(1))

    with sqlite3.connect(db_path) as connection:
        table = _table_with_columns(
            connection,
            {"table_a.__all_individual_income_tax_returns:_selected_income_and_tax_items_"},
        )
        if table is None:  # fall back to any table with "historical" in the name
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
            table = next((name for (name,) in tables if "historical" in name.lower()), None)
        if table is None:
            return None
        rows = connection.execute(f'SELECT * FROM "{table}"').fetchall()

    if len(rows) < 3:
        return None

    # Row index 2 holds the year headers; find which column is our year.
    year_columns = {}
    for index, value in enumerate(rows[2]):
        if index == 0 or value is None:
            continue
        try:
            normalized_year = int(float(value))
        except (TypeError, ValueError):
            continue
        if 1990 <= normalized_year <= 2023:
            year_columns[normalized_year] = index
    year_index = year_columns.get(tax_year)
    if year_index is None:
        return None

    # Among the metric rows, pick the label that best matches the question.
    query_tokens = set(tokenize(q))
    candidates = []
    for row in rows[5:]:
        if not isinstance(row[0], str) or not row[0].strip():
            continue
        label = row[0].strip()
        label_tokens = set(tokenize(label))
        if label_tokens and label_tokens.issubset(query_tokens):
            candidates.append((len(label_tokens), label, row[year_index]))

    if not candidates and "income tax returns" in q:
        for row in rows[5:]:
            if isinstance(row[0], str) and row[0].strip().lower() == "all returns":
                candidates.append((2, "All returns", row[year_index]))
                break
    if not candidates:
        return None

    _, label, value = max(candidates, key=lambda item: item[0])
    if value is None:
        return None
    try:
        number = float(str(value).replace(",", ""))
        value_text = f"{int(number):,}" if number.is_integer() else f"{number:,}"
    except (TypeError, ValueError):
        value_text = str(value)

    return {
        "answer": f"In {tax_year}, {label} was {value_text}.",
        "source": "SQLite",
        "document_number": _source_file(table),
        "metadata": {"table": table, "tax_year": tax_year, "item": label},
    }


# ===========================================================================
# PART 7 — THE ORCHESTRATOR: try each strategy in order
# ===========================================================================
def answer_question(query, store=None, top_k=5):
    # Strategy 1 & 2: exact answers from the database.
    for structured_answer in (sqlite_tax_answer(query), sqlite_historical_answer(query)):
        if structured_answer is not None:
            return [structured_answer]

    # Strategy 3: search the PDFs and return the best chunks.
    stores = [store] if store is not None else get_stores()
    results = []
    for current_store in stores:
        for hit in hybrid_search(current_store, query, top_k=top_k):
            hit["_store"] = current_store
            results.append(hit)

    results.sort(key=lambda r: r["hybrid_score"], reverse=True)
    results = results[:top_k]

    answers = []
    for hit in results:
        metadata = hit["metadata"]
        current_store = hit.pop("_store")
        answers.append({
            "answer": hit["text"],
            "source": "PDF",
            "document_number": metadata.get("source") or _pdf_document(current_store),
            "chunk_index": metadata.get("chunk_index", "unknown"),
            "metadata": metadata,
        })
    return answers


# ===========================================================================
# PART 8 — COMMAND-LINE INTERFACE
# ===========================================================================
def print_answers(query, answers):
    print(f"\nQuestion: {query}")
    if not answers:
        print("No answer found in SQLite or the IRS PDF.")
        return
    for index, result in enumerate(answers, start=1):
        clean_text = result["answer"][:1000].replace("\n", " ")
        print(f"\n[{index}] Answer: {clean_text}")
        print(f"    Source: {result['source']}")
        print(f"    Document: {result['document_number']}")
        if "chunk_index" in result:
            print(f"    Chunk index: {result['chunk_index']}")


def run_chat():
    """Interactive loop: keep asking for questions until the user quits."""
    while True:
        question = input("\nAsk a tax question (q to quit): ").strip()
        if question.lower() in {"q", "quit", "quite"}:
            print("Goodbye!")
            break
        if question:
            print_answers(question, answer_question(question))


if __name__ == "__main__":
    # If run with arguments, answer that one question; otherwise start the chat.
    question = " ".join(sys.argv[1:]).strip()
    if question:
        print_answers(question, answer_question(question))
    else:
        run_chat()