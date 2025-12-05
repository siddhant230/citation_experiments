import os
import pandas as pd
import json
from typing import Dict, Any

from src.data_source import generate_and_save_data_samples
from src.citation_methods.llm_based_citation import run_method_1_pipeline
from src.citation_methods.embedding_based_citation import run_method_2_pipeline
from src.visualize import visualize_results


def process_and_report_query(
    query_id: str,
    query: str,
    context_nodes: Dict[int, str],
    node_map: Dict[int, str],
    mode: str,
    test_goal: str,
    save_dir: str
) -> Dict[str, Any]:
    """
    Processes a single query using the specified mode, runs the pipeline, 
    visualizes results, and prepares a structured output.
    """
    print(f"\n=======================================================")
    print(f"       STARTING QUERY PROCESSING: {query_id}")
    print(f"MODE: {mode.upper()}")
    print(f"QUERY: {query}")
    print(f"=======================================================")

    # Run the full pipeline
    if mode.lower() == "llm":
        results = run_method_1_pipeline(query, context_nodes, node_map)
    elif mode.lower() == "embedding":
        results = run_method_2_pipeline(query, context_nodes, node_map)
    else:
        raise AssertionError(
            f"Wrong MODE selected: {mode}. Must be 'llm' or 'embedding'.")

    # Run visualization
    visualize_results(query_id=query_id, query=query,
                      results=results, node_map=node_map,
                      context_nodes=context_nodes,
                      filename=None,
                      save_html_path=save_dir)

    # Structure the output
    scenario_result = {
        "query_id": query_id,
        "query_text": query,
        "test_goal": test_goal,
        "cited_response": results['cited_response'],
        "contribution_stats": results['stats'],
        "profit_share": results['profit_share']
    }
    # Print summary for immediate review
    print("\n[LLM/SYNTHESIZED RESPONSE]")
    print(scenario_result['cited_response'])

    print("\n[PROFIT SHARE ALLOCATION (Normalized Score)]")
    # Convert to DataFrame for clean printing and sorting
    df = pd.DataFrame(scenario_result['profit_share'].items(), columns=[
        'Server', 'Share'])
    df = df.sort_values(by='Share', ascending=False).round(4)
    df['Share'] = df['Share'].apply(lambda x: f"{x:.4f}")
    print(df.to_string(index=False))
    print("-------------------------------------------------------")

    return scenario_result


if __name__ == "__main__":

    # -----------------------------------------------
    # 1. CONFIGURATION
    # -----------------------------------------------
    # Set the desired execution mode: 'llm' or 'embedding'
    # Change this variable to switch pipelines.
    SELECTED_MODE = "llm"  # "embedding"  # Try changing to "llm"

    filepath = generate_and_save_data_samples(save_path="simulation_data")

    # Define the output path for the final result file
    results_dir = f"results/{SELECTED_MODE}"
    os.makedirs(results_dir, exist_ok=True)

    # Load the scenario data from the JSON file
    with open(filepath, 'r') as f:
        scenarios = json.load(f)

    # -----------------------------------------------
    # 2. DEFINE INPUT DATA (Simulating user input)
    # -----------------------------------------------
    for idx, single_scenario in enumerate(scenarios):
        save_dir_scenario = f"{results_dir}/scenario_{idx}"
        os.makedirs(save_dir_scenario, exist_ok=True)

        # Extract components for the function call
        query_id = single_scenario['query_id']
        query = single_scenario['query_text']
        context_nodes = {
            int(k): v for k, v in single_scenario['context_nodes'].items()}
        node_map = {int(k): v for k, v in single_scenario['node_map'].items()}
        test_goal = single_scenario['test_goal']

        # -----------------------------------------------
        # 3. EXECUTION
        # -----------------------------------------------

        final_result = process_and_report_query(
            query_id=query_id,
            query=query,
            context_nodes=context_nodes,
            node_map=node_map,
            mode=SELECTED_MODE,
            test_goal=test_goal,
            save_dir=save_dir_scenario)

        # -----------------------------------------------
        # 4. FINAL SAVE
        # -----------------------------------------------

        try:

            final_output_path = f"{save_dir_scenario}/result.json"
            with open(final_output_path, 'w') as f:
                json.dump({"setup": single_scenario,
                           "result": final_result}, f, indent=4)

            print(
                f"\n✅ Query processed successfully using {SELECTED_MODE.upper()} mode.")
            print(f"Result saved to {final_output_path}")
        except Exception as e:
            print(f"\n--- ERROR: Could not save final output. Error: {e}")
