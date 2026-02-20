import os

import pytest

from config import load_config
from main import ml_pipeline

TEST_YAML_CONFIG_PATH: str = "tests/integration/test.yaml"


def create_temp_config_with_tmp_path(yaml_config_path, tmp_path):
    # Load the config YAML and replace <test_path> with actual tmp_path
    with open(yaml_config_path) as f:
        config_raw = f.read()
    config_str = config_raw.replace("<test_path>", str(tmp_path))

    # Write modified config to a temp file inside tmp_path
    temp_config_path = tmp_path / "temp_test_config.yaml"
    with open(temp_config_path, "w") as f:
        f.write(config_str)
    return temp_config_path


@pytest.mark.integration
class TestMLPipeline:
    def test_e2e_pipeline(self, tmp_path):
        # given
        test_config_path = create_temp_config_with_tmp_path(
            TEST_YAML_CONFIG_PATH, tmp_path
        )
        config = load_config(test_config_path)

        # when
        ml_pipeline(config)

        # then
        # Check for expected output files
        assert os.path.exists(config.data.dataset_file_path)
        assert os.path.exists(config.model.output_path)
        assert os.path.exists(config.model.output_params_path)
        assert os.path.exists(config.model.metrics_path)
