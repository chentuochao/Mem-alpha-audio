"""
Post-process an anonymized-speaker memory run: substitute inferred speaker names
into the stored memory so name-based QA retrieval works.

Background
----------
In the --anon_speaker experiment the memory agent sees <SpeakerX> tags (never the
real names) and:
  * keeps a Speaker registry in CORE memory (e.g. "Speaker0 = Sheldon [confirmed]"
    or the looser natural form "Speaker0 (Sheldon)"), and
  * stores every fact in semantic/episodic memory under the STABLE tag
    ("Speaker0 is a physicist").

Semantic/episodic memory is retrieved by embedding similarity to the question. A
question about "Sheldon" will NOT retrieve a fact worded "Speaker0 is a physicist"
because the surface forms don't match. Core memory holds the Speaker->name map but
carries no facts. So we must (a) rewrite the facts to use the real name AND
(b) re-embed them, or QA fails.

What this script does (per agent_state.json found under --agent_dir):
  1. Parse the Speaker->name map from state["core"] (the agent's OWN inferred
     names — keeps the experiment honest; no external oracle).
  2. Substitute SpeakerX -> name in every semantic/episodic item's content.
  3. Re-embed ONLY the changed items (OpenAI text-embedding-3-small) and update
     the matching rows of embeddings.npz.
  4. Rewrite agent_state.json + embeddings.npz, and dump speaker_name_map.json.
Core memory is left intact as the audit trail (and is injected wholesale at QA).

Usage:
  python postprocess_speaker_substitute.py --agent_dir <dir> [--dry_run]
    [--include_candidates] [--no_reembed]

  <dir> may be a single run folder (containing agent_state.json) or a parent; all
  agent_state.json files beneath it are processed.
"""

import argparse
import glob
import json
import os
import re

import numpy as np

UNKNOWN_TOKENS = {"unknown", "none", "n/a", "", "?"}


# ── Registry parsing ─────────────────────────────────────────────────────────

def parse_speaker_map(core: str, include_candidates: bool = True) -> dict[int, str]:
    """Extract {speaker_index -> name} from a core-memory string.

    Handles both the strict registry format written by the anon prompt
    ("Speaker0 = Sheldon [confirmed] (cues: ...)") and the looser natural form
    the model tends to produce ("Speaker0 (Sheldon)"). Strict entries win.
    """
    if not core:
        return {}

    speaker_map: dict[int, str] = {}

    # 1) Strict: "Speaker0 = Name [status]" (status + cues optional).
    #    Name runs until '[', '(', ',', newline, or end.
    for m in re.finditer(
        r"Speaker_?(\d+)\s*=\s*([^\[\](),\n]+?)\s*(?:\[(\w+)\])?\s*(?:\(|,|\n|$)",
        core,
    ):
        idx = int(m.group(1))
        name = m.group(2).strip()
        status = (m.group(3) or "").strip().lower()
        if name.lower() in UNKNOWN_TOKENS:
            continue
        if status == "unknown":
            continue
        if status == "candidate" and not include_candidates:
            continue
        speaker_map[idx] = name

    # 2) Natural: "Speaker0 (Sheldon)" — only fill gaps the strict pass missed.
    for m in re.finditer(r"Speaker_?(\d+)\s*\(([^)]+)\)", core):
        idx = int(m.group(1))
        if idx in speaker_map:
            continue
        name = m.group(2).strip()
        # Skip cue annotations like "(cues: ...)" and unknowns.
        if ":" in name or name.lower() in UNKNOWN_TOKENS:
            continue
        speaker_map[idx] = name

    return speaker_map


def substitute(text: str, speaker_map: dict[int, str]) -> str:
    """Replace whole-word SpeakerN / Speaker_N with the mapped name.

    Descending index order so "Speaker1" never clobbers part of "Speaker18"
    (word boundaries already guard this, but ordering makes it bulletproof).
    """
    for idx in sorted(speaker_map, reverse=True):
        name = speaker_map[idx]
        text = re.sub(rf"\bSpeaker_?{idx}\b", lambda _m: name, text)
    return text


# ── Embeddings ───────────────────────────────────────────────────────────────

def make_embedder():
    """Return a function text->np.ndarray using OpenAI, or None if unavailable."""
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        import openai
        client = openai.OpenAI()
    except Exception as e:
        print(f"  [embed] could not init OpenAI client: {e}")
        return None

    def _embed(text: str) -> np.ndarray:
        r = client.embeddings.create(model="text-embedding-3-small", input=text)
        return np.array(r.data[0].embedding)

    return _embed


# ── Per-run processing ───────────────────────────────────────────────────────

def process_state(state_path: str, include_candidates: bool, dry_run: bool,
                  no_reembed: bool) -> dict:
    with open(state_path, "r") as f:
        state = json.load(f)

    out_dir = os.path.dirname(state_path)
    core = state.get("core") or ""
    speaker_map = parse_speaker_map(core, include_candidates=include_candidates)

    summary = {
        "state_path": state_path,
        "n_speakers_mapped": len(speaker_map),
        "speaker_map": {f"Speaker{k}": v for k, v in sorted(speaker_map.items())},
        "semantic_changed": 0,
        "episodic_changed": 0,
        "reembedded": 0,
    }

    if not speaker_map:
        print(f"  [skip] no parseable speaker names in core: {state_path}")
        return summary

    # Substitute in semantic + episodic; remember which item ids changed.
    changed_ids = {"semantic": [], "episodic": []}
    for mtype in ("semantic", "episodic"):
        for item in state.get(mtype, []):
            for mem_id, content in list(item.items()):
                new_content = substitute(content, speaker_map)
                if new_content != content:
                    item[mem_id] = new_content
                    changed_ids[mtype].append(mem_id)
        summary[f"{mtype}_changed"] = len(changed_ids[mtype])

    if dry_run:
        print(f"  [dry_run] {os.path.basename(out_dir)}: "
              f"{summary['n_speakers_mapped']} names, "
              f"{summary['semantic_changed']} semantic / "
              f"{summary['episodic_changed']} episodic items would change")
        return summary

    # Re-embed changed items (rows aligned via *_embedding_ids).
    embedder = None if no_reembed else make_embedder()
    npz_path = os.path.join(out_dir, "embeddings.npz")
    if embedder is not None and os.path.exists(npz_path):
        npz = np.load(npz_path)
        matrices = {
            "semantic": npz["semantic_matrix"] if "semantic_matrix" in npz else np.empty((0, 1536)),
            "episodic": npz["episodic_matrix"] if "episodic_matrix" in npz else np.empty((0, 1536)),
        }
        content_by_id = {
            mtype: {mid: c for it in state.get(mtype, []) for mid, c in it.items()}
            for mtype in ("semantic", "episodic")
        }
        for mtype in ("semantic", "episodic"):
            ids = state.get(f"{mtype}_embedding_ids", [])
            mat = matrices[mtype]
            for mem_id in changed_ids[mtype]:
                if mem_id not in ids:
                    continue
                row = ids.index(mem_id)
                if row >= mat.shape[0]:
                    continue
                mat[row] = embedder(content_by_id[mtype][mem_id])
                summary["reembedded"] += 1
            matrices[mtype] = mat
        np.savez_compressed(npz_path,
                            semantic_matrix=matrices["semantic"],
                            episodic_matrix=matrices["episodic"])
    elif not no_reembed and embedder is None:
        print("  [warn] OPENAI_API_KEY unavailable -> text substituted but NOT "
              "re-embedded. Name-based retrieval will still miss until you "
              "re-embed. Re-run with the key set.")

    # Persist substituted state + the parsed map.
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    with open(os.path.join(out_dir, "speaker_name_map.json"), "w") as f:
        json.dump(summary["speaker_map"], f, indent=2, ensure_ascii=False)

    print(f"  [done] {os.path.basename(out_dir)}: {summary['n_speakers_mapped']} names, "
          f"changed {summary['semantic_changed']} sem / {summary['episodic_changed']} epi, "
          f"re-embedded {summary['reembedded']}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent_dir", required=True,
                        help="Run folder (with agent_state.json) or a parent; all "
                             "agent_state.json beneath it are processed.")
    parser.add_argument("--include_candidates", action="store_true", default=True,
                        help="Substitute [candidate] names too (default: on).")
    parser.add_argument("--confirmed_only", dest="include_candidates",
                        action="store_false",
                        help="Only substitute [confirmed] names; leave candidates as SpeakerX.")
    parser.add_argument("--no_reembed", action="store_true",
                        help="Substitute text but do not recompute embeddings.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Report the parsed map + change counts; write nothing.")
    args = parser.parse_args()

    if os.path.isfile(args.agent_dir) and args.agent_dir.endswith("agent_state.json"):
        state_paths = [args.agent_dir]
    else:
        state_paths = sorted(glob.glob(os.path.join(args.agent_dir, "**", "agent_state.json"),
                                       recursive=True))
        # also the direct case: <agent_dir>/agent_state.json
        direct = os.path.join(args.agent_dir, "agent_state.json")
        if os.path.exists(direct) and direct not in state_paths:
            state_paths.insert(0, direct)

    if not state_paths:
        print(f"No agent_state.json found under {args.agent_dir}")
        return

    print(f"Processing {len(state_paths)} state file(s) "
          f"(mode={'dry_run' if args.dry_run else 'write'}, "
          f"candidates={'yes' if args.include_candidates else 'no'})")
    for sp in state_paths:
        process_state(sp, include_candidates=args.include_candidates,
                      dry_run=args.dry_run, no_reembed=args.no_reembed)


if __name__ == "__main__":
    main()
