import re
from typing import Dict, Any, List
from collections import defaultdict


def aggregate_server_citations(llm_response: str, node_to_server_map: Dict[int, str]) -> Dict[str, Dict[str, Any]]:
    """Parses the LLM response and counts citations per server."""
    CITATION_PATTERN = r"<cite:\[(.*?)\]>"
    citation_groups = re.findall(CITATION_PATTERN, llm_response)

    CITATION_PATTERN = r"<cite:(\d+)>"
    citation_groups += re.findall(CITATION_PATTERN, llm_response)

    CITATION_PATTERN = r"<cite:([\d,]+)>"
    citation_groups += re.findall(CITATION_PATTERN, llm_response)

    server_citation_counts = defaultdict(int)
    server_multi_source_counts = defaultdict(int)
    for citation_str in citation_groups:
        try:
            node_indices: List[int] = [
                int(idx.strip()) for idx in citation_str.split(',') if idx.strip()]
        except ValueError:
            continue

        contributing_servers = set()
        for index in node_indices:
            server_name = node_to_server_map.get(index)
            if server_name:
                contributing_servers.add(server_name)

        is_multi_source = len(contributing_servers) > 1

        for server_name in contributing_servers:
            server_citation_counts[server_name] += 1
            if is_multi_source:
                server_multi_source_counts[server_name] += 1

    # Include all retrieved servers, even those with 0 citations
    all_servers = set(node_to_server_map.values())
    return {
        server: {
            'total_citations': server_citation_counts[server],
            'multi_source_citations': server_multi_source_counts[server]
        }
        for server in all_servers
    }


def calculate_profit_share(stats: Dict[str, Dict[str, Any]], k_multi: float = 1.5, k_single: float = 1.0) -> Dict[str, float]:
    """Computes normalized relevance score (profit share) based on weighted citations."""
    weightages: Dict[str, float] = {}

    for server, data in stats.items():
        multi = data.get('multi_source_citations', 0)
        total = data.get('total_citations', 0)
        single = total - multi

        # W_i = (k_multi * M_i) + (k_single * S_i)
        weightages[server] = (k_multi * multi) + (k_single * single)

    total_weight = sum(weightages.values())

    # R_i = W_i / sum(W_j)
    normalized_scores = {
        server: weight / total_weight if total_weight > 0 else 0.0
        for server, weight in weightages.items()
    }
    return normalized_scores
