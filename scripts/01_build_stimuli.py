#!/usr/bin/env python3
"""Build the S1-S6 synthetic stimulus set (research plan §4).

Writes ``data/stimuli/<CAT>/{audio,midi,manifest.jsonl,meta.json}``; S1/S2/S3
patching partners land in the S6 directory. Use ``--no-render`` to produce
MIDI + manifests only (no fluidsynth needed).
"""
from __future__ import annotations

import sys

from _common import data_root, make_parser, setup


def main() -> int:
    parser = make_parser(__doc__, "stimuli.yaml")
    parser.add_argument("--no-render", action="store_true",
                        help="skip audio rendering (MIDI + manifests only)")
    args = parser.parse_args()
    cfg = setup(args)

    from motif_circuits.stimuli import build_all
    from motif_circuits.utils.io import write_run_json

    out_root = data_root(cfg) / "stimuli"
    scfg = dict(cfg["stimuli"])
    render = bool(scfg.get("render", True)) and not args.no_render
    if (out_root / "S1" / "manifest.jsonl").is_file() and not args.force:
        print(f"{out_root} already populated; use --force to rebuild")
        return 0
    counts = build_all(scfg, out_root, render=render)
    write_run_json(out_root, cfg, {"counts": counts, "rendered": render})
    print(f"stimulus set at {out_root}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
