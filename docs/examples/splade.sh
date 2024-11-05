#!/bin/bash

model=$1

echo "Running model: $model"

echo "start"
date "+%Y-%m-%d %H:%M:%S"
echo ""

# Define eval_include variable
# full
eval_include="['mrtydi', 'jacwir', 'jagovfaqs_22k', 'nlp_journal_title_abs', 'nlp_journal_abs_intro', 'nlp_journal_title_intro', 'jaqket']"
# no mrtydi
# eval_include="['jacwir', 'jagovfaqs_22k', 'nlp_journal_title_abs', 'nlp_journal_abs_intro', 'nlp_journal_title_intro', 'jaqket']"

# fast
# eval_include="['jagovfaqs_22k', 'nlp_journal_title_abs', 'nlp_journal_abs_intro', 'nlp_journal_title_intro']"
# eval_include="['nlp_journal_title_abs', 'nlp_journal_abs_intro', 'nlp_journal_title_intro']"

poetry run python -m jmteb \
  --embedder SpladeEmbedder \
  --embedder.model_name_or_path "$model" \
  --embedder.device cuda \
  --save_dir "results/${model//\//_}" \
  --overwrite_cache false \
  --evaluators src/jmteb/configs/jmteb.jsonnet \
  --eval_include "$eval_include"

echo ""
date "+%Y-%m-%d %H:%M:%S"
echo "end"
