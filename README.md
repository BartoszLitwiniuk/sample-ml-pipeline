# Sample ML pipeline app (Python 3.12 + uv + Docker + LightGBM + optuna + Ruff)

A Python machine learning pipeline that downloads a dataset, prepares it with transformations and a train/test split, trains a LightGBM binary classifier using Optuna for hyperparameter tuning, and evaluates the model on the test set. It writes the trained model (LightGBM text format), the best hyperparameters, and evaluation metrics such as accuracy, precision, recall and F1. 

Parameters to the pipeline are passed using a config YAML file with data source URL and paths, train/test split, Optuna settings (number of trials and hyperparameter ranges), and model output paths.

**Requirements:**:
- Python 3.12, 
- [uv](https://github.com/astral-sh/uv),
- (optional) Docker.

---

## Setup

1. Clone repo, then: `make install-dev` or run in Docker (check below)
2. (Optional) Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`

---

## Project structure

```
config/base.yaml    # Pipeline config (data URL, paths, Optuna, model outputs)
src/
  main.py           # Entry: python src/main.py <config_path>
  config.py         # Load YAML → Config dataclasses
  data/             # Download (.rda→CSV), load, transform, train_test_split
  model/            # Optuna + LightGBM train/eval/save
  utils/            # seed, logger, YAML, metrics
tests/unit/         # Unit tests
tests/integration/  # Network / full-pipeline tests
infra/Dockerfile    # Multi-stage: base, test, runtime
Makefile            # install, install-dev, local, test, docker-build, docker-run, docker-test, lint, format, clean
```

---

## Run locally

```bash
make local
# or
uv run python src/main.py config/base.yaml
```

Outputs: 
- `data/output/model.txt`, 
- `data/output/params.json`, 
- `data/output/metrics.json`.

---

## Run in Docker

```bash
make docker-build
make docker-run
```

Custom config: `docker run --rm -v $(pwd)/config:/app/config ml-app config/base.yaml`

---

## Development
- Testing: `pytest` with coverage reporting
- Linting & formatting: [Ruff](https://docs.astral.sh/ruff/) — `make lint` (check), `make format` (reformat)
- Dependencies: Managed via `uv` and `pyproject.toml`

## Testing

| Where   | Command |
|--------|--------|
| Local  | `make test` |
| Docker | `make docker-test` |

---

## Usage & config

**CLI:** One argument — path to YAML config.

```bash
uv run python src/main.py config/base.yaml
```

Config sections: `random_state`, `data` (url, dataset_file_path, dataset_name), `dataset` (test_size), `optuna` (n_trials, hyperparameter ranges), `model` (output_path, output_params_path, metrics_path). See `config/base.yaml`.

---

## Makefile

| Target         | Description |
|----------------|-------------|
| `install` / `install-dev` | Dependencies |
| `local`        | Run pipeline (config/base.yaml) |
| `test`         | Pytest |
| `docker-build` / `docker-run` / `docker-test` | Docker |
| `lint` / `format` | Ruff |


### System Dependencies

#### macOS
LightGBM requires OpenMP library (`libomp`) on macOS. Install it using Homebrew:

```bash
brew install libomp
```
