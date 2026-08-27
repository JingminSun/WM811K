#!/usr/bin/env bash
#===================
# run.sh
#===================



#### Create a virtual environment if not available, based on requirements.txt
if [ ! -d .venv ]; then
    echo "=== creating .venv"
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
fi

source .venv/bin/activate

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"


mkdir -p outputs

### Preprocess the data if the data is not already done so.

if [ -f data/LSWMD_64.npz ]; then
    echo "=== preprocess: data/LSWMD_64.npz already exists, skipping"
else
    echo "=== preprocess"
    python3 -m utils.prepare_data \
        --pkl data/LSWMD.pkl \
        --output data/LSWMD_64.npz \
        --size 64 \
        2>&1 | tee outputs/prepare.log
fi


eval_only=false

mkdir -p outputs/

### Train and evaluate the model.
if [ "$eval_only" = false ]; then
    echo "=== train"
    python3 -m model.train 2>&1 | tee outputs/train.log
fi

echo "=== eval"
python3 -m model.eval 2>&1 | tee outputs/eval.log

echo "=== Completed. "