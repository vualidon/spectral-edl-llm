from __future__ import annotations

from skedl.cli import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "demo",
                "--backend",
                "mock",
                "--num-cots",
                "6",
                "--temp-mode",
                "mixed",
            ]
        )
    )
