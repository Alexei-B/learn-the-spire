"""Local web dashboard for training-run metrics (stdlib only, offline).

The trainer writes an append-only event stream per run under a directory
(default ``checkpoints/runs/<run_id>/``):

* ``manifest.json`` — ``{runId, label, startedAt, kind, argv, config, gitSha,
  featureVersion, catalogSignature}``.
* ``events.jsonl`` — one JSON object per line:
  ``{ts, phase, step, name, value, tags?}`` (``tags`` omitted when empty).

The dashboard treats the directory as its database: list runs = list dirs, live
= tail the files. Nothing in the trainer knows the dashboard exists; this package
only ever *reads* those files.

Run it with::

    python -m lts2_agent.dashboard --dir checkpoints/runs --port 8777

and generate a synthetic run to look at with::

    python -m lts2_agent.dashboard.demo --dir /tmp/runs --live
"""

from __future__ import annotations

from .store import RunStore, bucket_series

__all__ = ["RunStore", "bucket_series"]
