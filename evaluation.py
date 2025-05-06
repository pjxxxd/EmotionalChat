#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CBT-style per-utterance evaluator
--------------------------------
• Reads conversation JSON files from ./data/
• Sends each therapist reply (with full context) to an OpenAI model
• Receives seven numeric scores, computes per-turn and overall averages,
  and saves everything to ./results/
"""

import os
import glob
import json
import time
import statistics
import openai

# ───────────────────────────────────────────────────────────────
# 1.  Configuration
# ───────────────────────────────────────────────────────────────
openai.api_key = ""
MODEL_NAME     = "gpt-4o"                         # change if desired
RATE_LIMIT_SEC = 1.0                              # crude delay between calls

# ───────────────────────────────────────────────────────────────
# 2.  The evaluation rubric (system prompt) — FULL TEXT
# ───────────────────────────────────────────────────────────────
EVALUATION_PROMPT = r"""
# Role: CBT Dialogue Quality Evaluation Expert

## Goal:
Evaluate the CBT therapist's responses critically, rigorously, and objectively, based on an integrated framework adapted from established CBT evaluation frameworks. Use the following clearly defined dimensions:

### 1. **Language Fluency and Clarity**
Evaluate whether the therapist uses natural, clear, conversational language, avoiding jargon, awkward phrasing, or unnatural tone.

### 2. **Therapeutic Relevance and Focus**
Assess whether responses directly address the client's emotional and psychological concerns, explicitly reflecting the client's issues without irrelevant or tangential topics.

### 3. **CBT Role Consistency**  
Evaluate consistency in demonstrating core CBT therapeutic stances:
- **Collaborative Empiricism:** Actively involves the client in identifying and testing beliefs through joint exploration.
- **Structured Problem-Solving:** Clearly defines specific problems and systematically guides toward actionable solutions.
- **Socratic Questioning:** Uses reflective, open-ended questions to stimulate critical thinking and self-exploration.
- **Cognitive Restructuring:** Explicitly identifies and challenges negative automatic thoughts, core beliefs, and cognitive distortions.
- **Behavioral Strategies:** Suggests behavioral experiments or activities clearly designed to challenge dysfunctional thoughts or develop new behavioral patterns.

### 4. **CBT Knowledge and Accuracy**  
Evaluate explicit and accurate application of CBT techniques, specifically:
- **Psychoeducation:** Clearly explains cognitive-behavioral connections (thoughts, emotions, behaviors).
- **Identification of Cognitive Distortions:** Correctly labels cognitive distortions such as catastrophizing, personalization, overgeneralization, all-or-nothing thinking.
- **Behavioral Activation:** Clearly suggests activities that engage the client in positive or therapeutic behavior patterns.
- **Homework Assignments:** Assigns relevant, structured homework tasks (e.g., behavioral experiments, cognitive monitoring exercises) with clear rationale and instructions.
- **Evidence-Based Recommendations:** Offers empirically supported coping strategies relevant to client concerns.

### 5. **Structured and Logical Session Management**
Evaluate the therapist’s effectiveness in structuring the session, specifically:
- Clear initial agenda setting.
- Interim summarization of session points.
- Logical coherence and efficient transitions between session topics.

### 6. **Empathy, Emotional Validation, and Interpersonal Effectiveness**
Assess explicit expressions of empathy, emotional validation, and warm interpersonal engagement, including:
- Validating client emotions explicitly and authentically.
- Reflecting understanding and acceptance of client experiences.
- Using supportive, respectful, compassionate language while maintaining professional boundaries.

### 7. **Interactive Engagement and Collaboration**
Evaluate active therapist engagement that includes:
- Soliciting explicit client feedback regarding understanding and satisfaction.
- Encouraging client participation in solution development.
- Explicit assignment and follow-up on homework tasks.
- Providing ample opportunity for client input and questions.

## Scoring Criteria (each dimension scored 0–3， 0.5 is allowed):
- **0:** Poor or inappropriate performance; significant inaccuracies or misalignment; ineffective or irrelevant responses.
- **1:** Below acceptable standards; noticeable deficiencies or partial inaccuracies; limited effectiveness or partially relevant responses.
- **2:** Adequately meets standards; minor flaws or subtle inaccuracies present; generally effective and relevant.
- **3:** Exemplary performance; clearly meets or exceeds all CBT standards, effectively and accurately applied.

## Contextual Information:
- **Bot Name:** {bot.name}
- **Bot Personality:** CBT Therapist
- **Bot Description:** An experienced Cognitive Behavioral Therapist, trained in structured, collaborative, and evidence-based therapeutic conversations guided by core CBT principles.

## Current Scenario:
- **Relationship:** Therapist (bot) – Client (user)
- **Scene:** Virtual CBT therapeutic session

Format your scores clearly as numbers separated by spaces (e.g., "2 3 2 2 3 2").
"""
# ───────────────────────────────────────────────────────────────
# 3.  Helper: call the model and return list[float] of 7 scores
# ───────────────────────────────────────────────────────────────
def score_reply(full_convo: str, reply_text: str, idx: int) -> list[float]:
    """Send one therapist utterance for scoring and return seven floats."""
    user_msg = (
        "Here is the full conversation so far (UTF-8 JSON):\n\n"
        f"{full_convo}\n\n"
        "Evaluate **only** this therapist reply "
        f"(utterance index {idx}):\n\n"
        f"\"{reply_text}\"\n\n"
        "Return seven numbers as described."
    )

    response = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": EVALUATION_PROMPT},
            {"role": "user",   "content": user_msg}
        ],
        temperature=0
    )
    raw = response.choices[0].message.content.strip()
    numbers = [float(x) for x in raw.split()]
    if len(numbers) != 7:
        raise ValueError(f"Expected 7 numbers, got: '{raw}'")
    return numbers

# ───────────────────────────────────────────────────────────────
# 4.  Main batch-processing loop
# ───────────────────────────────────────────────────────────────
def main() -> None:
    os.makedirs("results", exist_ok=True)
    json_paths = glob.glob(os.path.join("data", "*.json"))
    if not json_paths:
        print("❌  No conversation files found in ./data/")
        return

    for path in json_paths:
        print(f"\n🗂️  Processing {os.path.basename(path)}")
        with open(path, encoding="utf-8") as f:
            convo = json.load(f)

        full_convo_str = json.dumps(convo, ensure_ascii=False, indent=2)
        per_turn = []
        dim_totals = [0.0] * 7  # accumulate per-dimension sums

        for idx, msg in enumerate(convo):
            if msg.get("role") != "therapist_cbt_prompt":
                continue
            try:
                scores = score_reply(full_convo_str, msg["content"], idx)
                avg_turn = statistics.mean(scores)
                dim_totals = [t + s for t, s in zip(dim_totals, scores)]

                per_turn.append({
                    "utterance_index": idx,
                    "therapist_reply": msg["content"],
                    "scores": scores,
                    "avg_turn_score": avg_turn
                })
                print(f"   • turn {idx:>3} → {scores} | avg {avg_turn:.2f}")
            except Exception as exc:
                print(f"   ! turn {idx} failed: {exc}")
            time.sleep(RATE_LIMIT_SEC)

        # ── write per-turn evaluations ──────────────────────────
        eval_out = os.path.join(
            "results",
            os.path.basename(path).replace(".json", "_evaluations.json")
        )
        with open(eval_out, "w", encoding="utf-8") as wf:
            json.dump(per_turn, wf, ensure_ascii=False, indent=2)
        print(f"   ✅  Saved per-turn evaluations → {eval_out}")

        # ── compute & write summary stats ──────────────────────
        if per_turn:
            num_turns = len(per_turn)
            overall_avg = statistics.mean(pt["avg_turn_score"] for pt in per_turn)
            per_dim_avg = [round(t / num_turns, 4) for t in dim_totals]

            summary = {
                "file": os.path.basename(path),
                "num_therapist_turns": num_turns,
                "overall_avg_score": round(overall_avg, 4),
                "per_dimension_avg": per_dim_avg
            }
            summary_out = os.path.join(
                "results",
                os.path.basename(path).replace(".json", "_summary.json")
            )
            with open(summary_out, "w", encoding="utf-8") as sf:
                json.dump(summary, sf, ensure_ascii=False, indent=2)
            print(f"   📊  Saved summary → {summary_out}")
        else:
            print("   (No therapist_cbt turns found)")

if __name__ == "__main__":
    main()

