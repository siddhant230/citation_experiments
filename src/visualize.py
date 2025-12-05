import os
from typing import Dict, List, Tuple
import pandas as pd
import re
from collections import defaultdict


def spool_color_map():
    pass


# --- VISUALIZATION CONSTANTS (Replicated) ---
# NOTE: Ensure all these constants are available in your main script
SERVER_COLOR_MAP: Dict[str, str] = {
    "Server A (Biology)": "#C7E9B4",   # Light Green
    "Server B (Critique)": "#FFC0CB",  # Pink
    "Server C (Poetry)": "#ADD8E6",    # Light Blue
    "Server M (Writer)": "#FFFACD",    # Lemon Yellow
    "Server Q (Energy)": "#F08080",    # Light Coral
    "Server D (Philosophical)": "#DDA0DD",  # Plum (Added for S1_Philosophical)
    "Server E (Physics)": "#B0E0E6",   # Powder Blue
    "Server X (Patents/Personnel)": "#FDFD96",  # Canary Yellow
    "Server Y (Marketing/Overview)": "#87CEFA",  # Light Sky Blue
    "Server Z (Geography)": "#98FB98",  # Pale Green
    "Server F (Legal Briefs)": "#FFD700",  # Gold
    "Server G (News Articles)": "#FFA07A",  # Light Salmon
    "Server H (Canadian Policy)": "#E6E6FA",  # Lavender
}
ANSI_RESET = '\033[0m'


# --- NEW VISUALIZATION CONSTANTS ---
MULTI_SERVER_COLOR = "repeating-linear-gradient(45deg, #f0f0f0, #f0f0f0 10px, #cccccc 10px, #cccccc 20px)"
# This creates a grey striped pattern to signify collaboration.

# --- Helper Functions (Stubs for completeness) ---


def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    """Converts a hex color code to an RGB tuple (stub)."""
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))


def get_ansi_bg_escape(hex_code: str) -> str:
    """Returns the ANSI escape code for 24-bit background color (stub)."""
    r, g, b = hex_to_rgb(hex_code)
    return f'\033[48;2;{r};{g};{b}m'


def generate_color_coded_response_html(
    cited_response: str,
    node_to_server_map: Dict[int, str]
) -> str:
    """
    Generates an HTML string of the response with cited text highlighted.
    Uses striped background for multi-server citations.
    """
    CITATION_PATTERN = r"(<cite:[\d,]+>)(.*?)(</cite>)"

    def replacer(match):
        citation_str = match.group(1)
        cited_text = match.group(2)

        citation_indices = re.findall(r'\d+', citation_str)
        node_indices = [int(idx) for idx in citation_indices]

        contributing_servers = {
            node_to_server_map.get(idx)
            for idx in node_indices
            if node_to_server_map.get(idx) in SERVER_COLOR_MAP
        }

        # Determine Color/Pattern
        server_names = sorted(list(contributing_servers))

        if len(server_names) > 1:
            # Multi-server: Use striped pattern
            style_value = f"background: {MULTI_SERVER_COLOR}"
            representative_server_name = server_names  # "Multi-Server Collaboration"
        elif len(server_names) == 1:
            # Single server: Use single color
            representative_server_name = server_names[0]
            color = SERVER_COLOR_MAP.get(representative_server_name)
            style_value = f"background-color: {color}"
        else:
            return cited_text

        # Tooltip text for the title attribute
        title_text = f"Cited from: {representative_server_name} (Nodes {', '.join(map(str, node_indices))})"

        return f'<span style="{style_value}; padding: 2px; border-radius: 3px; font-weight: bold; cursor: help;" title="{title_text}">{cited_text}</span>'

    colored_html = re.sub(CITATION_PATTERN, replacer, cited_response)

    return f'<div id="response-container" style="border: 1px solid #ddd; padding: 20px; background-color: #ffffff; line-height: 1.6; font-size: 16px; margin-bottom: 20px;">{colored_html}</div>'


def create_server_details_html(
    node_map: Dict[int, str],
    profit_share: Dict[str, float],
    context_nodes: Dict[int, str]
) -> str:
    """
    Generates an HTML accordion section showing per-server details: nodes, profit share.
    """
    # 1. Group nodes by server
    server_to_nodes: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for node_id, server_name in node_map.items():
        server_to_nodes[server_name].append(
            (node_id, context_nodes.get(node_id, "Content Not Found")))

    details_html = ""
    # Sort servers alphabetically for consistent display
    for server_name in sorted(server_to_nodes.keys()):
        node_list = server_to_nodes[server_name]
        share = profit_share.get(server_name, 0.0)
        color = SERVER_COLOR_MAP.get(server_name, "#f0f0f0")

        nodes_content = ""
        for node_id, content in node_list:
            nodes_content += f"""
            <div style="border-left: 2px solid {color}; padding-left: 10px; margin-top: 10px; background-color: #fff;">
                <strong>Node {node_id}:</strong> {content}
            </div>
            """

        details_html += f"""
        <details style="margin-bottom: 15px; border: 1px solid {color}; background-color: #ffffff;">
            <summary style="padding: 10px; cursor: pointer; background-color: {color}33; font-weight: bold;">
                {server_name} &mdash; Profit Share: {share:.4f}
            </summary>
            <div style="padding: 15px;">
                {nodes_content}
            </div>
        </details>
        """

    return f"""
    <div id="server-details-container" style="margin-top: 30px;">
        <h2>Detailed Server Contributions</h2>
        {details_html}
    </div>
    """


def generate_legend_html(node_map: Dict[int, str]) -> str:
    """Generates the HTML markup for the legend, including the multi-server stripe."""
    unique_servers = sorted(list(set(node_map.values())))

    legend_items = ""

    # 1. Add Multi-Server Stripe
    legend_items += f"""
    <div style="display: flex; align-items: center; margin-bottom: 8px;">
        <div style="width: 20px; height: 20px; background: {MULTI_SERVER_COLOR}; border: 1px solid #aaa; margin-right: 10px;"></div>
        <div style="font-weight: bold; color: #444;">Multi-Server Collaboration</div>
    </div>
    """

    # 2. Add Single Server Colors
    for server in unique_servers:
        color = SERVER_COLOR_MAP.get(server, "#f0f0f0")

        legend_items += f"""
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <div style="width: 20px; height: 20px; background-color: {color}; border: 1px solid #aaa; margin-right: 10px;"></div>
            <div style="font-weight: 500;">{server}</div>
        </div>
        """

    return f"""
    <div id="legend-container" style="border: 1px solid #ddd; padding: 15px; background-color: #f0f0f0; margin-top: 20px;">
        <h3 style="margin-top: 0; border-bottom: 1px solid #ccc; padding-bottom: 10px;">Server Contribution Legend (Single Source)</h3>
        {legend_items}
    </div>
    """


def create_full_html_report(query_id: str, query: str, response_html: str, legend_html: str, details_html: str) -> str:
    """Assembles the final, stand-alone HTML document."""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Attribution Report: {query_id}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 40px; background-color: #f4f4f9; }}
            h1 {{ color: #333; }}
            #query-box {{ background-color: #fff; padding: 15px; border-left: 5px solid #007bff; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>Server Attribution Report: {query_id}</h1>

        <div id="query-box">
            <p><strong>Query:</strong> {query}</p>
            <p><strong>Method:</strong> LLM-Based Citation (Method 1)</p>
        </div>

        <h2>Attributed Response</h2>
        {response_html}

        {legend_html}

        {details_html}

        <hr style="margin-top: 40px;">
        <p style="font-size: 0.8em; color: #777;">Visualization generated by the attribution pipeline. Hover over highlighted text for citation details.</p>
    </body>
    </html>
    """


def visualize_results(query_id, query, results, node_map, context_nodes,
                      filename=None, save_html_path="html_reports_method2"):
    os.makedirs(save_html_path, exist_ok=True)

    if filename is None:
        filename = f"{query_id}_attribution_report.html"
    file_path = f"{save_html_path}/{filename}"

    # 2. Generate HTML Components
    response_html = generate_color_coded_response_html(
        results['cited_response'], node_map)
    legend_html = generate_legend_html(node_map)
    details_html = create_server_details_html(
        node_map, results['profit_share'], context_nodes)  # NEW

    # 3. Assemble Full Report
    full_html_report = create_full_html_report(
        query_id, query, response_html, legend_html, details_html)  # Updated call

    # 4. Save HTML File

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(full_html_report)
