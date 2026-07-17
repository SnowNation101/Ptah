import argparse
import os
from dotenv import load_dotenv
import time

load_dotenv()

def get_api_key(url, env_var=None):
    """Determine the API key based on the URL or environment variable."""
    if env_var:
        return os.getenv(env_var, 'EMPTY')
    
    mapping = {
        'siliconflow': 'SILICONFLOW_API_KEY',
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY'
    }
    
    for key, var in mapping.items():
        if key in url.lower():
            return os.getenv(var, 'EMPTY')
            
    return 'EMPTY'

def main():
    parser = argparse.ArgumentParser(description="Run Ptah pipeline.")
    parser.add_argument("--base_url", type=str, required=True, help="Base URL for the LLM API.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name to use.")
    parser.add_argument("--reviewer_base_url", type=str, required=True, help="Base URL for the reviewer LLM API.")
    parser.add_argument("--reviewer_model_name", type=str, required=True, help="Reviewer model name to use.")
    parser.add_argument("--task", type=str, required=True, choices=["custom", "drb", "dc"], help="Task to run.")
    parser.add_argument("--question", type=str, help="Question for custom task.")
    parser.add_argument(
        "--section_workers",
        type=int,
        default=1,
        help="Number of report sections to research concurrently. Use 1 for serial execution.",
    )
    parser.add_argument("--start_id", type=int, help="First benchmark item id to run.")
    parser.add_argument("--end_id", type=int, help="Last benchmark item id to run.")
    parser.add_argument("--limit", type=int, help="Maximum number of benchmark items to run.")
    parser.add_argument(
        "--disable_html_refine",
        action="store_true",
        help="Build the HTML preview without the final screenshot-based HTML refinement step.",
    )
    parser.add_argument(
        "--current_date",
        type=str,
        help="Optional date string injected into every model call, e.g. 2026-07-15.",
    )
    args = parser.parse_args()

    if args.current_date:
        os.environ["PTAH_CURRENT_DATE"] = args.current_date

    args.api_key = get_api_key(args.base_url)
    args.reviewer_api_key = get_api_key(args.reviewer_base_url)

    start_time = time.time()
    if args.task == "custom":
        from tasks.custom_task import CustomTask
        task = CustomTask(args)
        task.run()
    elif args.task == "drb":
        from tasks.drb_task import DRBTask
        task = DRBTask(args)
        task.run()
    elif args.task == "dc":
        from tasks.dc_task import DCTask
        task = DCTask(args)
        task.run()

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print("\n\n" + "=" * 50)
    print(f"Execution time: {int(minutes)} m {int(seconds):02} s")
    print("=" * 50 + "\n\n")


if __name__ == "__main__":
    main()
