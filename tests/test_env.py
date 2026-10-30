"""Unit tests for environment variable loader."""

import os
from lifemate.utils.env import load_env_file


def test_load_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# Test comment\n"
        "TEST_ENV_VAR=hello_world\n"
        'TEST_QUOTED_VAR="quoted_value"\n'
        "TEST_SINGLE_QUOTED='single_value'\n"
        "\n"
        "INVALID_LINE\n",
        encoding="utf-8",
    )

    loaded = load_env_file(env_file, override=True)
    assert loaded["TEST_ENV_VAR"] == "hello_world"
    assert loaded["TEST_QUOTED_VAR"] == "quoted_value"
    assert loaded["TEST_SINGLE_QUOTED"] == "single_value"
    assert os.environ.get("TEST_ENV_VAR") == "hello_world"
    assert os.environ.get("TEST_QUOTED_VAR") == "quoted_value"
    assert os.environ.get("TEST_SINGLE_QUOTED") == "single_value"

