"""Launch the training dashboard: ``python -m lts2_agent.dashboard``."""

from __future__ import annotations

import argparse
import os

from .server import make_server


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m lts2_agent.dashboard",
        description="Local web dashboard over training-run event files (stdlib only).",
    )
    parser.add_argument(
        "--dir",
        default="checkpoints/runs",
        help="Directory holding one subdirectory per run (default: checkpoints/runs).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8777, help="Bind port (default: 8777).")
    args = parser.parse_args()

    root = os.path.abspath(args.dir)
    os.makedirs(root, exist_ok=True)  # tolerate the runs dir not existing yet

    httpd = make_server(root, args.host, args.port)
    host, port = httpd.server_address[:2]
    print(f"Lts2 training dashboard: http://{host}:{port}  (runs dir: {root})", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
