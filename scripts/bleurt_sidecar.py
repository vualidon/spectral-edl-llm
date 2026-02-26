#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BLEURT scoring sidecar for SK-EDL TruthfulQA runs")
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        from bleurt import score as bleurt_score
    except Exception as exc:
        sys.stderr.write(f"failed to import BLEURT: {exc}\n")
        return 2

    scorer = bleurt_score.BleurtScorer(checkpoint=args.checkpoint)

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
            references = [str(x) for x in (req.get("references") or [])]
            candidates = [str(x) for x in (req.get("candidates") or [])]
            scores = scorer.score(references=references, candidates=candidates)
            out = {"scores": [float(s) for s in (scores or [])]}
        except Exception as exc:
            out = {"error": f"{exc.__class__.__name__}: {exc}"}
            traceback.print_exc(file=sys.stderr)
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
