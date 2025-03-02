#!/bin/bash

# Exit on error
set -e

models=("xgboost")

folds=(0 1 2 3 4)

for model in "${models[@]}"; do
  for fold in "${folds[@]}"; do
    echo "Running: python train.py --fold $fold --model $model"
    if ! python3 train.py --fold $fold --model $model; then
      echo "Error: Failed to train $model on fold $fold"
      exit 1
    fi
  done
done
