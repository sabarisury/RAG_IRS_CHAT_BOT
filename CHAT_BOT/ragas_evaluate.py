"""
ragas_evaluate.py — evaluate the tax RAG pipeline with the RAGAS library.

RAGAS scores things the keyword eval can't. For each question it needs four
fields, which this script assembles from your own pipeline:

    user_input          the question                     (from eval_set.json)
    retrieved_contexts  the chunks retrieval pulled       (from answer_question)
    response            the bot's generated answer        (from TaxChatbot.chat)
    reference           the gold answer                   (from eval_set.json)

Metrics computed:
    Faithfulness              — is the answer grounded in the retrieved context
                                (i.e. not hallucinated)?
    ResponseRelevancy         — does the answer actually address the question?
    LLMContextPrecision...    — were the retrieved chunks relevant / well-ranked?
    LLMContextRecall          — did retrieval bring back what the reference needs?

All four use an LLM (and embeddings) as judge, so this needs your OPENAI_API_KEY
and will make API calls — one small batch per metric per question.

Run it:
    uv run python ragas_evaluate.py
    uv run python ragas_evaluate.py --limit 5     # quick, cheaper trial run

Requires:  uv add ragas langchain-openai datasets
"""

import argparse
import json
import warnings
from pathlib import Path

from dotenv import load_dotenv

# RAGAS 0.4.x prints deprecation notices for these stable import paths; the
# newer collections API isn't a drop-in yet, so keep these and hush the noise.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

# Your pipeline
from retrival import answer_question
from IRS import TaxChatbot

# RAGAS + the judge models
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
EVAL_SET_PATH = BASE_DIR / "eval_set.json"
RESULTS_CSV = BASE_DIR / "ragas_results.csv"

TOP_K = 5
JUDGE_MODEL = "gpt-4o-mini"       # the LLM that grades each metric
EMBED_MODEL = "text-embedding-3-small"  # used by relevancy/precision


def build_samples(eval_set, top_k=TOP_K):
    """
    For each question, run the real pipeline and collect the four fields
    RAGAS expects. A fresh bot per question keeps the turns independent
    (no conversation history leaking between eval items).
    """
    samples = []
    for item in eval_set:
        question = item["question"]

        # Retrieval → the contexts actually used.
        results = answer_question(question, top_k=top_k)
        contexts = [r.get("answer", "") for r in results] or ["(no context retrieved)"]

        # Generation → the answer the user would see.
        response = TaxChatbot(top_k=top_k).chat(question)

        samples.append({
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": response,
            "reference": item.get("reference_answer", ""),
        })
    return samples


def main():
    parser = argparse.ArgumentParser(description="RAGAS evaluation for the tax RAG pipeline.")
    parser.add_argument("--limit", type=int, default=None, help="only evaluate the first N questions")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="contexts to retrieve per question")
    parser.add_argument("--data", type=Path, default=EVAL_SET_PATH, help="path to the eval JSON")
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    if args.limit:
        eval_set = eval_set[: args.limit]

    print(f"Preparing {len(eval_set)} samples (running retrieval + generation)...")
    samples = build_samples(eval_set, top_k=args.top_k)
    dataset = EvaluationDataset.from_list(samples)

    # Wrap the OpenAI models so RAGAS can use them as judges.
    judge_llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=EMBED_MODEL))

    metrics = [
        Faithfulness(llm=judge_llm),
        ResponseRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        LLMContextPrecisionWithReference(llm=judge_llm),
        LLMContextRecall(llm=judge_llm),
    ]

    print("Scoring with RAGAS (this makes API calls and may take a few minutes)...\n")
    result = evaluate(dataset=dataset, metrics=metrics)

    # Aggregate scores across all questions.
    print("=" * 60)
    print("RAGAS SUMMARY")
    print("=" * 60)
    for metric_name, score in result._repr_dict.items() if hasattr(result, "_repr_dict") else []:
        print(f"   {metric_name:<28} {score:.3f}")
    # Fallback / always print the standard view too.
    print("\n", result, "\n")

    # Per-question breakdown → CSV.
    df = result.to_pandas()
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Saved per-question scores to {RESULTS_CSV.name}")


if __name__ == "__main__":
    main()