"""
evaluate.py — measure how well the RAG pipeline retrieves and answers.

It scores two layers against a small gold dataset (eval_set.json):

  RETRIEVAL  — did the right source document come back, and how high?
      * Hit Rate@k : fraction of questions whose expected source was retrieved
      * MRR        : mean reciprocal rank of the first correct source (1.0 = top)

  ANSWER     — is the returned text actually correct?
      * Keyword recall : fraction of expected keywords present in the answer
      * Answer accuracy : fraction of questions that clear a keyword threshold
      * (optional) LLM judge : gpt-4o-mini rates each answer PASS/FAIL

Run it standalone:
    uv run python evaluate.py                # keyword-based metrics only
    uv run python evaluate.py --judge        # also run the LLM judge (uses API)

The dataset is a JSON list; each item may define any of:
    question           (required)
    expected_keywords  phrases a correct answer should contain
    expected_source    filename the answer should be grounded in (or null)
    reference_answer    a short gold answer, used only by the LLM judge
"""

import argparse
import csv
import json
from pathlib import Path

from retrival import answer_question

BASE_DIR = Path(__file__).resolve().parent
EVAL_SET_PATH = BASE_DIR / "eval_set.json"
RESULTS_CSV = BASE_DIR / "eval_results.csv"

TOP_K = 5                    # retrieve this many results per question
KEYWORD_THRESHOLD = 0.5      # answer is "accurate" if >= this share of keywords hit


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def load_eval_set(path=EVAL_SET_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Metric helpers — each takes one question's data + the retrieved results
# ---------------------------------------------------------------------------
def reciprocal_rank(results, expected_source):
    """1 / (rank of the first result from the expected source). 0 if never found."""
    if not expected_source:
        return None  # this question doesn't test retrieval
    for rank, result in enumerate(results, start=1):
        if result.get("document_number") == expected_source:
            return 1.0 / rank
    return 0.0


def hit_at_k(results, expected_source):
    """1 if the expected source appears anywhere in the results, else 0."""
    if not expected_source:
        return None
    sources = {r.get("document_number") for r in results if isinstance(r, dict)}
    return 1.0 if expected_source in sources else 0.0


def keyword_recall(answer_text, expected_keywords):
    """Fraction of expected keywords found (case-insensitive) in the answer."""
    if not expected_keywords:
        return None
    text = answer_text.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in text)
    return hits / len(expected_keywords)


# ---------------------------------------------------------------------------
# Optional LLM-as-judge (only imported/used when --judge is passed)
# ---------------------------------------------------------------------------
def make_judge():
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def judge(question, answer, reference):
        prompt = (
            "You are grading a tax chatbot. Decide if the ANSWER correctly and "
            "relevantly addresses the QUESTION, using the REFERENCE as a guide to "
            "what a correct answer covers. Reply with exactly one word: PASS or FAIL.\n\n"
            f"QUESTION: {question}\n"
            f"REFERENCE: {reference}\n"
            f"ANSWER: {answer}\n\n"
            "Verdict:"
        )
        verdict = llm.invoke(prompt).content.strip().upper()
        return 1.0 if verdict.startswith("PASS") else 0.0

    return judge


# ---------------------------------------------------------------------------
# Run the evaluation
# ---------------------------------------------------------------------------
def evaluate(eval_set, top_k=TOP_K, use_judge=False):
    judge = make_judge() if use_judge else None
    rows = []

    for item in eval_set:
        question = item["question"]
        results = answer_question(question, top_k=top_k)
        top = results[0] if results else {}
        top_answer = top.get("answer", "")

        row = {
            "question": question,
            "hit@k": hit_at_k(results, item.get("expected_source")),
            "reciprocal_rank": reciprocal_rank(results, item.get("expected_source")),
            "keyword_recall": keyword_recall(top_answer, item.get("expected_keywords")),
            "top_source": top.get("document_number", "(none)"),
            "answer_preview": top_answer[:120].replace("\n", " "),
        }
        recall = row["keyword_recall"]
        row["answer_accurate"] = None if recall is None else float(recall >= KEYWORD_THRESHOLD)

        if judge is not None:
            row["llm_pass"] = judge(question, top_answer, item.get("reference_answer", ""))

        rows.append(row)

    return rows


def _mean(values):
    """Average the non-None values; None if there are none to average."""
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def summarize(rows):
    def col(name):
        return _mean([r.get(name) for r in rows])

    summary = {
        "questions": len(rows),
        "hit_rate@k": col("hit@k"),
        "mrr": col("reciprocal_rank"),
        "keyword_recall": col("keyword_recall"),
        "answer_accuracy": col("answer_accurate"),
    }
    if any("llm_pass" in r for r in rows):
        summary["llm_pass_rate"] = col("llm_pass")
    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt(value):
    return "  n/a " if value is None else f"{value:6.2f}"


def print_report(rows, summary):
    print("\n" + "=" * 78)
    print("PER-QUESTION RESULTS")
    print("=" * 78)
    for r in rows:
        print(f"\nQ: {r['question']}")
        print(f"   hit@k={_fmt(r['hit@k'])}  rr={_fmt(r['reciprocal_rank'])}"
              f"  kw_recall={_fmt(r['keyword_recall'])}"
              f"  accurate={_fmt(r['answer_accurate'])}"
              + (f"  llm={_fmt(r.get('llm_pass'))}" if "llm_pass" in r else ""))
        print(f"   source: {r['top_source']}")
        print(f"   answer: {r['answer_preview']}")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for key, value in summary.items():
        label = key.replace("_", " ").title()
        print(f"   {label:<20} {value if isinstance(value, int) else _fmt(value)}")
    print()


def save_csv(rows, path=RESULTS_CSV):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-question results to {path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate the tax RAG pipeline.")
    parser.add_argument("--judge", action="store_true", help="also run the LLM judge (uses the OpenAI API)")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="results to retrieve per question")
    parser.add_argument("--data", type=Path, default=EVAL_SET_PATH, help="path to the eval JSON")
    args = parser.parse_args()

    eval_set = load_eval_set(args.data)
    rows = evaluate(eval_set, top_k=args.top_k, use_judge=args.judge)
    summary = summarize(rows)
    print_report(rows, summary)
    save_csv(rows)


if __name__ == "__main__":
    main()