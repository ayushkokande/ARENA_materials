#!/bin/bash
# Executes the 4.1 notebook headlessly on the GPU pod.
# Expects: model-organisms-for-EM cloned into chapter4_alignment_science/exercises/,
#          a .env there with OPENROUTER_API_KEY=<anything> (the notebook asserts it is set),
#          and judge_server.py reachable on 127.0.0.1:8765 (e.g. `ssh -R 8765:127.0.0.1:8765 <pod>`).
cd "$(dirname "$0")/.."
export JUDGE_BASE_URL=http://127.0.0.1:8765/v1 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
date +"START %T"
jupyter nbconvert --to notebook --execute 4.1_Emergent_Misalignment.ipynb \
  --output 4.1_Emergent_Misalignment_executed.ipynb \
  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=2400
echo "EXIT $?"; date +"END %T"
