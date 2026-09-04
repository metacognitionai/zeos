# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""Settings resolve as: .env file < environment variables < flags."""

from zeos_space_invaders.utils import config

#: Captured at import, before conftest's `no_dotenv` replaces it for every test.
#: This is the one file that wants the real lookup.
find_env_file = config.find_env_file


def test_a_value_already_in_the_environment_wins(tmp_path, monkeypatch):
    """The precedence rule: the file must not overwrite a real variable."""
    env = tmp_path / ".env"
    env.write_text("OPENAI_MODEL=from-the-file\n")
    monkeypatch.setenv("OPENAI_MODEL", "from-the-environment")
    config.load_env(env)
    assert config.os.environ["OPENAI_MODEL"] == "from-the-environment"


def test_a_value_only_in_the_file_is_loaded(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OPENAI_MODEL=from-the-file\n")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert config.load_env(env) == env
    assert config.os.environ["OPENAI_MODEL"] == "from-the-file"


def test_an_empty_value_means_not_configured(tmp_path, monkeypatch):
    """`.env.example` ships valueless keys; they must not blank a real one."""
    env = tmp_path / ".env"
    env.write_text("OPENAI_TOP_K=\n")
    monkeypatch.setenv("OPENAI_TOP_K", "20")
    config.load_env(env)
    assert config.os.environ["OPENAI_TOP_K"] == "20"


def test_comments_blanks_and_junk_lines_are_skipped(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# a comment\n\nNOT_AN_ASSIGNMENT\nA_KEY = 'quoted value'\n")
    monkeypatch.delenv("A_KEY", raising=False)
    config.load_env(env)
    assert config.os.environ["A_KEY"] == "quoted value"


def test_no_env_file_anywhere_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    assert find_env_file() is None
    assert config.load_env() is None


def test_the_working_directory_is_preferred_over_the_checkout(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("A=1\n")
    monkeypatch.chdir(tmp_path)
    assert find_env_file() == tmp_path / ".env"
