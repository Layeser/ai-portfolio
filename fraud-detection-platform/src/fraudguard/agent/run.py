"""Command line entry point for the investigation agent.

    python -m fraudguard.agent.run --tx-id T0531065       # one alert
    python -m fraudguard.agent.run --top 3                # highest-score alerts
    python -m fraudguard.agent.run --evaluate 20          # compare verdicts with ground truth
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import pandas as pd
from dotenv import load_dotenv

from fraudguard.config import ARTIFACTS_DIR, REPORTS_DIR, SEED


def render_markdown(decision: dict, tools_used: list[str]) -> str:
    r = decision["report"]
    lines = [
        f"### Alert {r['tx_id']} — model score {decision['model_score']:.3f}",
        "",
        f"**Verdict:** `{r['verdict']}` ({r['confidence']} confidence)  ",
        f"**Recommended action:** `{r['recommended_action']}`  ",
        f"**Human approval required:** {'yes' if decision['requires_human_approval'] else 'no'}",
        "",
        r["summary"],
        "",
        "**Evidence**",
        *[f"- {e}" for e in r["evidence"]],
    ]
    if decision.get("untraced_values"):
        values = ", ".join(decision["untraced_values"])
        lines += ["", f"**Values to verify (not found in tool outputs):** {values}"]
    if decision["policy_notes"]:
        lines += ["", "**Policy guardrails applied**", *[f"- {n}" for n in decision["policy_notes"]]]
    lines += ["", f"_Tools called: {', '.join(tools_used)}_"]
    return "\n".join(lines)


async def investigate(tx_ids: list[str]) -> list[dict]:
    from fraudguard.agent.graph import build_graph

    graph = await build_graph()
    results = []
    for i, tx_id in enumerate(tx_ids, 1):
        start = time.perf_counter()
        for attempt in range(3):
            try:
                state = await graph.ainvoke({"tx_id": tx_id})
                break
            except Exception as exc:  # network / rate-limit errors: retry, then skip the alert
                print(f"[{tx_id}] attempt {attempt + 1} failed: {type(exc).__name__}", flush=True)
                await asyncio.sleep(10 * (attempt + 1))
        else:
            continue
        decision = state["decision"].model_dump()
        results.append({"tx_id": tx_id, "decision": decision, "tools_used": state["tools_used"],
                        "seconds": round(time.perf_counter() - start, 1)})
        print(f"[{i}/{len(tx_ids)}] {tx_id}: {decision['report']['verdict']}", flush=True)
    return results


def evaluation_sample(n: int) -> list[str]:
    """Hard cases first: every fraud in the ambiguous 'review' band, then legit reviews and blocks."""
    alerts = pd.read_parquet(ARTIFACTS_DIR / "alerts.parquet")
    review = alerts[alerts.decision == "review"]
    parts = [
        review[review.is_fraud == 1],
        review[review.is_fraud == 0].sample(n // 2, random_state=SEED),
        alerts[alerts.decision == "block"].sample(n // 4, random_state=SEED),
    ]
    return pd.concat(parts).head(n)["tx_id"].tolist()


def evaluate(results: list[dict]) -> dict:
    labels = pd.read_parquet(ARTIFACTS_DIR / "alerts.parquet").set_index("tx_id")
    rows = []
    for r in results:
        truth = int(labels.loc[r["tx_id"], "is_fraud"])
        verdict = r["decision"]["report"]["verdict"]
        rows.append({"tx_id": r["tx_id"], "band": labels.loc[r["tx_id"], "decision"], "is_fraud": truth,
                     "verdict": verdict, "action": r["decision"]["report"]["recommended_action"]})
    df = pd.DataFrame(rows)
    decided = df[df.verdict != "needs_human_review"]
    correct = ((decided.verdict == "likely_fraud") == (decided.is_fraud == 1)).sum()
    missed = df[(df.is_fraud == 1) & (df.verdict == "likely_legitimate")]
    return {
        "n_alerts": len(df),
        "frauds_in_sample": int(df.is_fraud.sum()),
        "auto_decided": len(decided),
        "accuracy_when_decided": round(correct / max(len(decided), 1), 3),
        "frauds_cleared_by_mistake": len(missed),
        "legit_alerts_closed": int(((df.is_fraud == 0) & (df.verdict == "likely_legitimate")).sum()),
        "escalated_to_human": int((df.verdict == "needs_human_review").sum()),
        "rows": rows,
    }


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tx-id")
    group.add_argument("--top", type=int)
    group.add_argument("--evaluate", type=int)
    args = parser.parse_args()

    if args.tx_id:
        tx_ids = [args.tx_id]
    elif args.top:
        from fraudguard.agent.case_store import CaseStore

        tx_ids = [a["tx_id"] for a in CaseStore().list_open_alerts(args.top)]
    else:
        tx_ids = evaluation_sample(args.evaluate)

    results = asyncio.run(investigate(tx_ids))
    out_dir = REPORTS_DIR / "agent"
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        md = render_markdown(r["decision"], r["tools_used"])
        (out_dir / f"{r['tx_id']}.md").write_text(md + "\n")
        print(md, f"\n({r['seconds']} s)\n")
    if args.evaluate:
        summary = evaluate(results)
        (out_dir / "evaluation.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
