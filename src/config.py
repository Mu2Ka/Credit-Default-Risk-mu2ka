from pathlib import Path

import yaml


PARAMS_PATH = Path(__file__).resolve().parent.parent / "configs/params.yaml"
with open(PARAMS_PATH, encoding="utf-8") as file:
    PARAMS = yaml.safe_load(file)
