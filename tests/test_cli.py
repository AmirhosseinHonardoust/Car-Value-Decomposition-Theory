from __future__ import annotations

from pathlib import Path

import pytest

from src import cli


@pytest.mark.parametrize("subcommand", ["prepare-data", "train", "evaluate", "decode-car"])
def test_build_parser_accepts_each_subcommand(subcommand: str) -> None:
    parser = cli.build_parser()
    args = parser.parse_args([subcommand])
    assert callable(args.func)


def test_build_parser_rejects_unknown_subcommand() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["not-a-real-command"])


def test_cli_prepare_data_end_to_end(raw_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["prepare-data"])
    args.func(args)
    out = capsys.readouterr().out
    assert "Prepared clean dataset" in out


def test_cli_train_then_evaluate_then_decode(
    raw_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = cli.build_parser()

    args = parser.parse_args(["train"])
    args.func(args)
    assert "Saved model" in capsys.readouterr().out

    args = parser.parse_args(["evaluate"])
    args.func(args)
    assert "Evaluation metrics" in capsys.readouterr().out

    args = parser.parse_args(["decode-car", "--index", "0"])
    args.func(args)
    assert "Value decomposition" in capsys.readouterr().out


def test_cli_decode_car_out_of_range_raises(raw_csv: Path) -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["train"])
    args.func(args)

    args = parser.parse_args(["decode-car", "--index", "9999"])
    with pytest.raises(IndexError):
        args.func(args)
