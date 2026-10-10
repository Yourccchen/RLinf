"""Train an RL Token compressor from exported frozen Soma features."""

import argparse
import json
from pathlib import Path

from rlinf.serving.rlt_features import train_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path, help="JSON constructor fields")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    losses = train_encoder(
        sorted(args.samples.glob("*.pt")), args.output,
        json.loads(args.model_config.read_text()), steps=args.steps,
        batch_size=args.batch_size, lr=args.lr, device=args.device, seed=args.seed,
    )
    print(json.dumps({"steps": len(losses), "first_loss": losses[0], "last_loss": losses[-1],
                      "checkpoint": str(args.output)}))


if __name__ == "__main__":
    main()
