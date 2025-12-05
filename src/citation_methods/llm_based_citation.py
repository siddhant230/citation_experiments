import os
import dspy
from dspy.signatures import Signature
from typing import Dict

from src.utils import aggregate_server_citations, calculate_profit_share

from dotenv import load_dotenv
load_dotenv()


# Fetch the API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"


# 1. Initialize the Language Model (LLM) using dspy.LM
try:
    lm = dspy.LM(model=MODEL_NAME, api_key=GEMINI_API_KEY)
    dspy.configure(lm=lm)
    print(f"DSPy configured with {MODEL_NAME}.")
except Exception as e:
    # Print the error message to help with debugging
    print(f"Error configuring DSPy with Gemini: {e}")
    print("Please ensure your GEMINI_API_KEY is correct and the dspy library is up-to-date.")


class GenerateAnswerWithCitations(Signature):
    """
    Generate a concise answer to the query using ONLY the provided sources.
    You MUST wrap every factual phrase, quoted text, or synthesized idea with
    the exact citation format: <cite:[NODE_INDICES]></cite>, where NODE_INDICES
    is a comma-separated list of the source indices (e.g., 1,4,6).
    Every sentence or claim must be inside a <cite> tag.
    Do not include any text outside a <cite> tag.
    """
    sources: str = dspy.InputField(
        desc="Numbered source content (e.g., '1: content 1; 2: content 2')")
    query: str = dspy.InputField()
    answer: str = dspy.OutputField(
        desc="Answer containing only <cite:[indices]>...</cite> tags. with an extra summary section and closure section")


class RAGCiter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_cited_answer = dspy.Predict(GenerateAnswerWithCitations)

    def forward(self, query: str, context: Dict[int, str]) -> str:
        # Format context into a single, numbered string for the prompt
        sources_str = "\n".join([f"{k}: {v}" for k, v in context.items()])

        # Call the LLM (Gemini) via the DSPy Predict module
        prediction = self.generate_cited_answer(
            sources=sources_str, query=query)
        return prediction.answer


def run_method_1_pipeline(query: str, context: Dict[int, str], node_map: Dict[int, str]) -> Dict[str, float]:
    """Runs the full pipeline: LLM generation -> Parsing -> Profit Share."""

    rag_citer = RAGCiter()

    try:
        cited_response = rag_citer(query=query, context=context)
    except Exception as e:
        print(f"LLM Call FAILED. Using MOCK response. Error: {e}")
        # Re-run the mock logic here to ensure consistency if the call fails
        cited_response = rag_citer.lm(
            query=query, sources="Mock sources").answer

    # 2. Citation Aggregation
    stats = aggregate_server_citations(cited_response, node_map)

    # 3. Profit Share Calculation (k_multi=1.5, k_single=1.0)
    profit_share = calculate_profit_share(stats)

    return {
        "cited_response": cited_response,
        "stats": stats,
        "profit_share": profit_share
    }
