"""SKMEM-P0-18: the voice memory search always declares a reader."""

from unittest.mock import MagicMock, patch

from skvoice.memory import search_memories


def _run(returncode=0, stdout="[long-term] abc.. | note | x"):
    return MagicMock(returncode=returncode, stdout=stdout)


def test_search_passes_public_reader_by_default():
    with patch("skvoice.memory.subprocess.run", return_value=_run()) as run:
        assert search_memories("what is for dinner", "lumina")
    cmd = run.call_args[0][0]
    assert cmd[cmd.index("--reader") + 1] == "@public"


def test_search_passes_declared_reader():
    with patch("skvoice.memory.subprocess.run", return_value=_run()) as run:
        search_memories("what is for dinner", "lumina", reader="channel:room1")
    cmd = run.call_args[0][0]
    assert cmd[cmd.index("--reader") + 1] == "channel:room1"


def test_blank_reader_is_public_never_owner():
    with patch("skvoice.memory.subprocess.run", return_value=_run()) as run:
        search_memories("what is for dinner", "lumina", reader="  ")
    cmd = run.call_args[0][0]
    assert cmd[cmd.index("--reader") + 1] == "@public"


def test_old_cli_without_reader_returns_nothing():
    # click exits 2 on an unknown option; the old CLI never runs the search.
    with patch("skvoice.memory.subprocess.run", return_value=_run(2, "")):
        assert search_memories("what is for dinner", "lumina") == ""
