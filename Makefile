.PHONY: install install-dev lock run test lint format docker-build docker-run docker-test clean

# Variables
DOCKERFILE_DIR := infra
APP_NAME := ml-app
TEST_IMAGE := $(APP_NAME)-test
CONFIG_PATH := config/base.yaml

# Local
install: 
	uv sync

install-dev:
	uv sync --group dev

lock:
	uv lock

test:
	uv run --group dev pytest -v

lint:
	uv run --group dev ruff check .

format:
	uv run --group dev ruff format .

local:
	uv run python src/main.py $(CONFIG_PATH)

# Docker targets
docker-build:
	docker build -f $(DOCKERFILE_DIR)/Dockerfile -t $(APP_NAME) .

docker-run:
	docker run --rm $(APP_NAME) $(CONFIG_PATH)

docker-test:
	docker build -f $(DOCKERFILE_DIR)/Dockerfile --target test -t $(TEST_IMAGE) .
	docker run --rm $(TEST_IMAGE)


clean:
	rm -rf .venv
	rm -f uv.lock