#!/usr/bin/env python3
# python3 init_weights.py --config-name-or-path configs/config_60m.json --hub-model-id PrimeIntellect/llama-60m-fresh
from transformers import AutoConfig, AutoModelForCausalLM
from cyclopts import App
import os

app = App()


@app.default
def main(
    config_name_or_path: str,
    hub_model_id: str | None = None,
    save_to_disk: str | None = None,
):
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path=config_name_or_path)
    from transformers import LlamaForCausalLM

    model: LlamaForCausalLM = AutoModelForCausalLM.from_config(config)
    print(model)
    if save_to_disk:
        os.makedirs(save_to_disk, exist_ok=True)
        model.save_pretrained(save_to_disk)
    if hub_model_id:
        model.push_to_hub(hub_model_id)


if __name__ == "__main__":
    app()
