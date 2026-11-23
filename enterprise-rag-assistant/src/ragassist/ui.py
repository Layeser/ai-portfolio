"""Gradio chat UI:  python -m ragassist.ui  ->  http://127.0.0.1:7860"""

from __future__ import annotations

import gradio as gr

from ragassist.pipeline import build_pipeline

EXAMPLES = [
    "Quel est le plafond d'un repas pendant une mission à Madrid ?",
    "Mon PC de boulot a disparu dans le métro, je dois prévenir qui ?",
    "Combien de jours de RTT a un salarié non cadre ?",
    "Quel est le montant de la prime de fin d'année ?",
]


def main() -> None:
    pipeline = build_pipeline()

    def respond(message: str, history: list):
        result = pipeline.ask(message)
        if result.citations:
            sources = "\n".join(f"- **[{c.n}]** {c.doc} — {c.section}, p.{c.page}" for c in result.citations)
            body = f"{result.answer}\n\n**Sources**\n{sources}"
        else:
            body = result.answer
        latency = result.timings_ms.get("total") or sum(result.timings_ms.values())
        return f"{body}\n\n<sub>{latency:.0f} ms · {result.refusal_reason or 'answered'}</sub>"

    gr.ChatInterface(
        respond,
        title="Atelier Nord — assistant des politiques internes",
        description="Réponses sourcées issues des documents RH, finance, IT et juridique. "
                    "Le système refuse de répondre quand l'information n'est pas dans les documents.",
        examples=EXAMPLES,
    ).launch()


if __name__ == "__main__":
    main()
