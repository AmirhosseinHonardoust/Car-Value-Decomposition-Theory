from __future__ import annotations

import argparse

from . import config, data_prep
from .evaluate_model import evaluate_model
from .train_model import train_price_model
from .value_decomposition import decompose_value_for_row


def cmd_prepare_data(args: argparse.Namespace) -> None:
    df = data_prep.load_clean()
    print(f"Prepared clean dataset with {len(df)} rows.")


def cmd_train(args: argparse.Namespace) -> None:
    train_price_model()


def cmd_evaluate(args: argparse.Namespace) -> None:
    evaluate_model()


def cmd_decode_car(args: argparse.Namespace) -> None:
    df = data_prep.load_clean()
    idx = args.index

    if idx < 0 or idx >= len(df):
        raise IndexError(f"Index {idx} is out of range for dataset of size {len(df)}.")

    row = df.iloc[idx]
    dec = decompose_value_for_row(row)

    print("Car specs:")
    cols_to_show = [
        config.COL_CAR_NAME,
        config.COL_YEAR,
        config.COL_FUEL_TYPE,
        config.COL_SELLER_TYPE,
        config.COL_TRANSMISSION,
        config.COL_KMS_DRIVEN,
        config.COL_OWNER,
        config.COL_PRESENT_PRICE,
    ]
    for c in cols_to_show:
        if c in row.index:
            print(f"- {c}: {row[c]}")

    print("\nValue decomposition:")
    for k, v in dec.items():
        print(f"{k}: {v:.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Car-Value-Decoding-Engine CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_prepare = subparsers.add_parser(
        "prepare-data", help="Clean and cache the dataset."
    )
    p_prepare.set_defaults(func=cmd_prepare_data)

    p_train = subparsers.add_parser("train", help="Train the price model.")
    p_train.set_defaults(func=cmd_train)

    p_eval = subparsers.add_parser("evaluate", help="Evaluate the price model.")
    p_eval.set_defaults(func=cmd_evaluate)

    p_decode = subparsers.add_parser(
        "decode-car", help="Decode a car's price into value components."
    )
    p_decode.add_argument(
        "--index",
        type=int,
        default=0,
        help="Row index from the cleaned dataset to decode.",
    )
    p_decode.set_defaults(func=cmd_decode_car)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
