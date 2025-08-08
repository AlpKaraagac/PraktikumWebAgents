#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

api_key=""     # Gemini/OpenAI key
export GOOGLE_API_KEY=""
export OPENAI_API_KEY=""

jsonl_path="auto-optimizer/example/WebJudge_Online_Mind2Web_eval_gemini-2.5-flash_score_threshold_3_auto_eval_results.json"

echo
echo ">>> 1) Categorizing traces"
python auto-optimizer/analyze_results.py \
  --jsonl "${jsonl_path}" \
  --out "auto-optimizer/example/failure_corpus.md"

echo
echo ">>> 2) Sumarizing traces"
python auto-optimizer/summarize_categories_failures.py \
  --corpus_md "auto-optimizer/example/failure_corpus.md" \
  --out "auto-optimizer/example/category_summaries.md"

echo
echo ">>> 3) Generating auto-optimization prompts"
python auto-optimizer/auto_prompt_optimizer.py \
    --prompt_file "browser-use/browser_use/agent/system_prompt.md" \
    --summary_md "auto-optimizer/example/category_summaries.md" \
    --out "auto-optimizer/example/new_system_prompt.md" \