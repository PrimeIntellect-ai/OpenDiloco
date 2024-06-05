#! /usr/bin/env python3
# Example: ./scripts/pull-model.py PrimeIntellect/llama-1b-fresh
import sys
from transformers import AutoModelForCausalLM

MODEL = sys.argv[1] if len(sys.argv) >= 2 else "PrimeIntellect/llama-1b-fresh"
model = AutoModelForCausalLM.from_pretrained(MODEL)

print(model)
