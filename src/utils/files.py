from pathlib import Path

import yaml


def read_yaml(path: str):
    file_path = Path(path)

    if not file_path.exists():
        raise Exception(f"File not found: {file_path}")

    try:
        with file_path.open("r") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise Exception(f"Invalid YAML: {e}") from e
