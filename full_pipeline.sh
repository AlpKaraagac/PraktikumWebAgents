#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

api_key=""     # Gemini/OpenAI key
model_name="gemini-2.5-flash"
temperature="0.2"

start_id=""                                           # e.g. "92160852a6bb..."

score_threshold=3
num_worker=1

export GOOGLE_API_KEY=""
export OPENAI_API_KEY=""

for i in $(seq 1 3); do
  output_dir="./trajectories/tr${i}"            # top-level dir for this run
  base_dir="${output_dir}"              # where run.py will look
  result_prefix="WebJudge_Online_Mind2Web_eval_${model_name}_score_threshold_${score_threshold}_auto_eval_results.json"
  jsonl_path="${base_dir}_result/${result_prefix}"
  csv_output="${base_dir}_result/failed_tasks.csv"

  echo
  echo ">>> 1) Generating traces with run_online_mw2.py"
  python ./run_online_mw2.py \
    --start_id       "${start_id}" \
    --output_dir     "${output_dir}" \
    --model          "${model_name}" \
    --temperature    "${temperature}" 

  echo
  echo ">>> 2) Auto-evaluating traces"
  python Online-Mind2Web/src/run.py \
    --model            "${model_name}" \
    --trajectories_dir "${base_dir}" \
    --api_key          "${api_key}" \
    --output_path      "${base_dir}" \
    --error_path       "${base_dir}" \
    --num_worker       "${num_worker}" \
    --score_threshold  "${score_threshold}"

  echo
  echo "  Run $i complete:"
  echo "     JSONL : $jsonl_path"
  echo "     CSV   : $csv_output"
done

echo
echo "  All runs finished.  Check the output_dir directories for outputs."
