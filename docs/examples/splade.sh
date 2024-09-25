model=$1

echo "Running model: $model"

echo "start"
date "+%Y-%m-%d %H:%M:%S"
echo ""

poetry run python -m jmteb \
  --embedder SpladeEmbedder \
  --embedder.model_name_or_path "$model" \
  --embedder.device cuda \
  --save_dir "results/${model//\//_}" \
  --overwrite_cache false \
  --evaluators src/jmteb/configs/jmteb.jsonnet \
  --eval_include "['mrtydi', 'jagovfaqs_22k','nlp_journal_title_abs', 'nlp_journal_abs_intro', 'nlp_journal_title_intro', 'jaqket', 'esci']"
#  --eval_include "['jagovfaqs_22k','nlp_journal_title_abs', 'nlp_journal_abs_intro', 'nlp_journal_title_intro', 'jaqket', 'esci']"
#   --eval_include "['jaqket', 'jagovfaqs_22k', 'mrtydi']"

#   --eval_include "['nlp_journal_title_abs']"

#  --eval_include "['nlp_journal_title_abs', 'nlp_journal_abs_intro', 'nlp_journal_title_intro']"

#  --eval_include "['jaqket', 'nlp_journal_title_abs']"


echo ""
date "+%Y-%m-%d %H:%M:%S"
echo "end"