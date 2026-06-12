"""Command-line interface for the garev pipeline.

Subcommands:
    generate-sample   Write a synthetic, raw-shaped dataset for demos and tests.
    train             Fit the two-stage model and save it to disk.
    evaluate          Train and report metrics for one timeframe.
    predict           Score visitors with a saved model and write a submission.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib

from garev.config import PipelineConfig
from garev.data.sample import write_sample
from garev.models.evaluate import evaluate as evaluate_model
from garev.pipeline import load_sessions, make_timeframe, predict_submission, train

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("garev")


def _cmd_generate_sample(args: argparse.Namespace) -> None:
    path = write_sample(Path(args.out), n_visitors=args.visitors, seed=args.seed)
    logger.info("Wrote synthetic dataset to %s", path)


def _cmd_train(args: argparse.Namespace) -> None:
    config = PipelineConfig()
    sessions = load_sessions(args.data, config)
    dataset = make_timeframe(sessions, config)
    trained = train(dataset, config)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained, out)
    logger.info("Trained on %d visitors; model saved to %s", len(dataset.features), out)


def _cmd_evaluate(args: argparse.Namespace) -> None:
    config = PipelineConfig()
    sessions = load_sessions(args.data, config)
    dataset = make_timeframe(sessions, config)
    trained = train(dataset, config)
    report = evaluate_model(
        trained.model, trained._align(dataset.features), dataset.is_buyer, dataset.log_revenue
    )
    for key, value in report.as_dict().items():
        logger.info("%-28s %s", key, value)


def _cmd_predict(args: argparse.Namespace) -> None:
    trained = joblib.load(args.model)
    sessions = load_sessions(args.data, PipelineConfig())
    submission = predict_submission(trained, sessions)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out, index=False)
    logger.info("Wrote %d predictions to %s", len(submission), out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="garev", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-sample", help="write a synthetic dataset")
    gen.add_argument("--out", default="data/sample.csv")
    gen.add_argument("--visitors", type=int, default=800)
    gen.add_argument("--seed", type=int, default=13)
    gen.set_defaults(func=_cmd_generate_sample)

    tr = sub.add_parser("train", help="fit and save the model")
    tr.add_argument("--data", required=True)
    tr.add_argument("--out", default="artifacts/model.joblib")
    tr.set_defaults(func=_cmd_train)

    ev = sub.add_parser("evaluate", help="train and report metrics")
    ev.add_argument("--data", required=True)
    ev.set_defaults(func=_cmd_evaluate)

    pr = sub.add_parser("predict", help="score visitors with a saved model")
    pr.add_argument("--data", required=True)
    pr.add_argument("--model", default="artifacts/model.joblib")
    pr.add_argument("--out", default="submission.csv")
    pr.set_defaults(func=_cmd_predict)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
