#!/usr/bin/env python3
import argparse
import json
import ast
import sys
import csv

def main():
    parser = argparse.ArgumentParser(
        description="Analyze JSONL of task records: print unsuccessful task_ids, summary stats, and dump failed tasks to CSV."
    )
    parser.add_argument(
        "json_file",
        help="Path to the JSON lines file (one dict per line)."
    )
    parser.add_argument(
        "--csv-output",
        default="failed_tasks.csv",
        help="Path to output CSV for failed tasks (default: failed_tasks.csv)."
    )
    args = parser.parse_args()

    total_tasks = 0
    successful_tasks = 0
    step_counts = {}
    failed_tasks = []

    try:
        with open(args.json_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Parse the top-level JSON object
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # Skip invalid JSON lines
                    continue

                total_tasks += 1
                task_id = record.get("task_id")
                task_desc = record.get("task", "")
                # Count steps from image_judge_record length if present
                img_judge = record.get("image_judge_record", [])
                step_counts[task_id] = len(img_judge)

                # Determine success/failure via predicted_label
                # Assume top-level 'predicted_label' exists
                pred_label = record.get("predicted_label")
                if pred_label == 1:
                    successful_tasks += 1
                else:
                    # for any non-1 (including 0 or missing), treat as failure
                    failed_tasks.append({
                        "task_id": task_id,
                        "task": task_desc,
                        "image_judge_record": img_judge,
                        "response": record.get("evaluation_details", {}).get("response", "")
                    })

        # Print all unsuccessful task_ids
        print("Unsuccessful task_ids:")
        for ft in failed_tasks:
            print(ft["task_id"])

        # Summary statistics
        print("\nSummary:")
        print(f"Total tasks: {total_tasks}")
        print(f"Successful tasks: {successful_tasks}")
        if total_tasks > 0:
            success_rate = successful_tasks / total_tasks * 100
            print(f"Success rate: {success_rate:.2f}%")
            avg_steps = sum(step_counts.values()) / total_tasks
            print(f"Average steps per task: {avg_steps:.2f}")

        # Write CSV for failed tasks
        with open(args.csv_output, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["task_id", "task", "image_judge_record", "response"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for ft in failed_tasks:
                # Serialize image_judge_record as JSON string
                writer.writerow({
                    "task_id": ft["task_id"],
                    "task": ft["task"],
                    "image_judge_record": json.dumps(ft["image_judge_record"], ensure_ascii=False),
                    "response": ft["response"]
                })

        print(f"\nFailed task details written to CSV: {args.csv_output}")

    except FileNotFoundError:
        sys.stderr.write(f"Error: File not found: {args.json_file}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"Unexpected error: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()