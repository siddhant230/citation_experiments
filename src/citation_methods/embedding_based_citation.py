from src.utils import aggregate_server_citations, calculate_profit_share
import os
import torch
import re
import nltk
import pandas as pd
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types  # Import the types module

from dotenv import load_dotenv
load_dotenv()


nltk.download('punkt')
nltk.download('punkt_tab')

# --- 0. Configuration and Utilities ---

# Load the BGE-Small model globally (can be slow on first run)
EMBEDDING_MODEL = SentenceTransformer('BAAI/bge-small-en-v1.5')
SIMILARITY_THRESHOLD = 0.75  # Threshold for declaring alignment
N_GRAM_SIZE = 5            # Use 5-word chunks

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"

# Initialize the Gemini Client
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
    print(f"Gemini client initialized for model {MODEL_NAME}.")
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    # Exit or handle error if the client cannot be initialized
    # For this example, we'll continue with the placeholder text below.
    client = None


# --- NEW FUNCTION: Replace mock with actual Gemini call ---

def generate_llm_response(query: str, context: Dict[int, str]) -> str:
    if client is None:
        # Placeholder response if client initialization failed
        return f"LLM client error. Placeholder response for query: {query}"

    # Format the context into a clean, numbered list for the prompt
    context_list = "\n".join([f"Node {k}: {v}" for k, v in context.items()])

    prompt = f"""
    You are an expert answer generator. Based ONLY on the following context,
    provide a concise and synthesized answer to the user's question.
    Do NOT mention the node indices or cite your sources.
    Do NOT introduce any external knowledge.

    --- CONTEXT ---
    {context_list}
    ---

    USER QUESTION: {query}

    CONCISE ANSWER:
    """

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,  # Keep temperature low for factual accuracy
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API call failed for query '{query}': {e}")
        return f"API FAILED: Could not generate response for query: {query}"


def split_text_into_chunks(text: str, chunk_size: int) -> List[str]:
    """Splits text into overlapping word n-gram chunks."""
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    chunks = []
    if len(words) < chunk_size:
        return [" ".join(words)]

    for i in range(len(words) - chunk_size + 1):
        chunks.append(" ".join(words[i: i + chunk_size]))
    return chunks


def auto_cite_response(
    response: str,
    context: Dict[int, str],
    server_map: Dict[int, str]
) -> str:
    """
    Simulates the LLM's citation process by finding the best source for each 
    response n-gram and builds a <cite:nodes> string, preserving original formatting.
    """

    # --- Step 1: Alignment Map (Identical to previous logic) ---
    # This step determines which node (if any) is the best match for each response n-gram.

    response_chunks = split_text_into_chunks(response, N_GRAM_SIZE)
    if not response_chunks:
        return response

    # Flatten context for embedding: (sentence, node_index, server_name)
    source_triples: List[Tuple[str, int, str]] = []
    for node_idx, node_content in context.items():
        # --- HERE IT IS LOOKED UP ---
        server_name = server_map.get(node_idx, "UNKNOWN")
        source_sentences = nltk.sent_tokenize(node_content)
        for sentence in source_sentences:
            source_triples.append((sentence, node_idx, server_name))

    source_chunks = [t[0] for t in source_triples]
    if not source_chunks:
        return response

    # Generate Embeddings and Similarity (Assuming EMBEDDING_MODEL is defined)
    all_texts = response_chunks + source_chunks
    with torch.no_grad():
        embeddings = EMBEDDING_MODEL.encode(
            all_texts, convert_to_tensor=True, show_progress_bar=False)

    response_embeddings = embeddings[:len(response_chunks)]
    source_embeddings = embeddings[len(response_chunks):]
    similarity_matrix = cosine_similarity(
        response_embeddings.cpu(), source_embeddings.cpu())

    # Map Chunk Index (index into response_chunks) -> Best Node Index (int)
    chunk_to_node_map: List[int] = []

    for i in range(len(response_chunks)):
        chunk_sims = similarity_matrix[i]
        best_match_idx = torch.argmax(torch.tensor(chunk_sims)).item()
        best_similarity = chunk_sims[best_match_idx]

        if best_similarity >= SIMILARITY_THRESHOLD:
            chunk_to_node_map.append(source_triples[best_match_idx][1])
        else:
            chunk_to_node_map.append(0)

    # --- Step 2: Reconstruct with Citations (Preserving Spaces) ---

    # Tokenize the original response while preserving spaces and punctuation
    # Example: "The woods are lovely." -> ['The', ' ', 'woods', ' ', 'are', ' ', 'lovely', '.']

    # Simple, non-destructive tokenization that keeps spaces and punctuation as separate tokens
    tokens = re.findall(r'(\s+)|([^\s]+)', response)
    # The result is a list of tuples like [('', 'The'), (' ', ''), ('', 'woods'), ...]

    original_tokens: List[str] = [t[0] or t[1] for t in tokens]

    # Create a parallel map of citation IDs for every *word* token.
    # This requires mapping the n-gram index back to the word index.

    # Only the actual words/punctuation
    word_tokens: List[str] = [t[1] for t in tokens if t[1]]
    sanitized_words = [re.sub(r'[^\w]', '', w).lower(
    ) for w in word_tokens if w and re.match(r'\w+', w)]  # Words only for indexing

    word_to_node_map: List[int] = [0] * len(sanitized_words)

    for chunk_idx, node_id in enumerate(chunk_to_node_map):
        if node_id != 0:
            # Propagate the node ID to all words covered by this n-gram
            start_word_idx = chunk_idx  # Assuming n-gram starts at the corresponding word index
            for offset in range(N_GRAM_SIZE):
                word_idx = start_word_idx + offset
                if word_idx < len(sanitized_words):
                    # Use the LAST node ID found to claim a word for continuity
                    word_to_node_map[word_idx] = node_id

    cited_response_parts: List[str] = []
    current_citation_node = 0
    current_cited_segment = ""

    # Traverse the original tokens (words, spaces, punctuation)
    sanitized_word_index = 0

    for token in original_tokens:
        is_word = bool(re.match(r'\w+', token))

        if is_word:
            # Look up citation ID for this word
            mapped_node = word_to_node_map[sanitized_word_index] if sanitized_word_index < len(
                word_to_node_map) else 0
            sanitized_word_index += 1
        else:
            # Punctuation or spaces inherit the citation ID of the preceding word (if any)
            mapped_node = current_citation_node

        # Check for Citation Break
        if mapped_node != current_citation_node:
            # 1. Close the previous segment
            if current_citation_node != 0:
                cited_response_parts.append(
                    f"<cite:{current_citation_node}>{current_cited_segment}</cite>")
            else:
                cited_response_parts.append(current_cited_segment)

            # 2. Start new segment
            current_citation_node = mapped_node
            current_cited_segment = token
        else:
            # Continue current segment
            current_cited_segment += token

    # Close the final segment
    if current_citation_node != 0:
        cited_response_parts.append(
            f"<cite:{current_citation_node}>{current_cited_segment}</cite>")
    else:
        cited_response_parts.append(current_cited_segment)

    return "".join(cited_response_parts)


def run_method_2_pipeline(query: str, context: Dict[int, str], node_map: Dict[int, str]):
    """
    Runs the Method 2 pipeline (LLM -> Alignment -> Auto-Cite) 
    and returns Method 1-compatible output.
    """

    # 1. LLM Generation
    response_text = generate_llm_response(query, context)

    # 2. Alignment and Auto-Citation (The core Method 2 visualization step)
    cited_response = auto_cite_response(response_text, context, node_map)

    # 3. Citation Aggregation (Reusing Method 1 logic)
    stats = aggregate_server_citations(cited_response, node_map)

    # 4. Profit Share Calculation (Reusing Method 1 logic)
    # NOTE: You'll need to define k_multi and k_single globally or pass them here.
    profit_share = calculate_profit_share(stats)

    return {
        "cited_response": cited_response,  # Now compatible with Method 1 viz
        "stats": stats,
        "profit_share": profit_share
    }
