"""Unit tests for utils.files."""

import pytest

from utils.files import read_yaml


class TestReadYaml:
    """Tests for read_yaml."""

    def test_read_yaml_returns_parsed_content(self, tmp_path):
        """read_yaml returns dict from valid YAML file."""
        # given
        path = tmp_path / "config.yaml"
        path.write_text("key: value\nnumber: 42\n")
        # when
        result = read_yaml(str(path))
        # then
        assert result == {"key": "value", "number": 42}

    def test_read_yaml_nested_structure(self, tmp_path):
        """read_yaml handles nested YAML structures."""
        # given
        path = tmp_path / "nested.yaml"
        path.write_text("a:\n  b: 1\n  c: [x, y]\n")
        # when
        result = read_yaml(str(path))
        # then
        assert result["a"]["b"] == 1
        assert result["a"]["c"] == ["x", "y"]

    def test_read_yaml_file_not_found_raises(self):
        """read_yaml raises when file does not exist."""
        # when
        with pytest.raises(Exception) as exc_info:
            read_yaml("/nonexistent/path/config.yaml")
        # then
        assert "File not found" in str(exc_info.value)

    def test_read_yaml_invalid_yaml_raises(self, tmp_path):
        """read_yaml raises when YAML is invalid."""
        # given
        path = tmp_path / "bad.yaml"
        path.write_text("key: [unclosed\n")
        # when
        with pytest.raises(Exception) as exc_info:
            read_yaml(str(path))
        # then
        assert "Invalid YAML" in str(exc_info.value)
