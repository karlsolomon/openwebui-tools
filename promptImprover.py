#!/usr/bin/env python3
# prompt_improver_stdlib.py
# -------------------------------------------------------------
# A tiny “prompt‑improver” pipe that works with only the std‑lib.
# -------------------------------------------------------------

import json
import logging
import sys
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

# ----------------------------------------------------------------------
# Logging – keep it simple but configurable
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
class State(Enum):
    WAITING_FOR_PROMPT = 0
    WAITING_FOR_ANSWERS = 1


@dataclass
class SessionState:
    original_prompt: str = ""
    current_prompt: str = ""
    questions_list: List[Dict] = field(default_factory=list)
    answers_list: List[Dict] = field(default_factory=list)
    state: State = State.WAITING_FOR_PROMPT


# ----------------------------------------------------------------------
# Helper for HTTP calls (replaces ``requests``)
# ----------------------------------------------------------------------
def _post_json(url: str, payload: dict, timeout: int = 30) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


# ----------------------------------------------------------------------
# Core pipe implementation
# ----------------------------------------------------------------------
class Pipe:
    # ------------------------------------------------------------------
    # Configuration – can be overridden at runtime if needed
    # ------------------------------------------------------------------
    class Valves:
        PROMPT_IMPROVER_MODEL = "chatGPT:latest"
        OLLAMA_BASE_URL = "http://192.168.4.26:11434/v1/chat/completions"

    def __init__(self):
        self.valves = self.Valves()
        self._session = SessionState()

    # ------------------------------------------------------------------
    # Extract the latest user message from an OpenAI‑style payload
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_user_input(body: dict) -> str:
        msgs = body.get("messages", [])
        for msg in reversed(msgs):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    # ------------------------------------------------------------------
    # Main entry point – receives the whole request body (dict)
    # ------------------------------------------------------------------
    def pipe(self, body: dict) -> str:
        # prepend a system message so the model knows the current state
        system_msg = {
            "role": "system",
            "content": f"You are in State {self._session.state}.",
        }
        body.setdefault("messages", []).insert(0, system_msg)

        # --------------------------------------------------------------
        # 1️⃣ First‑time handling – waiting for the original prompt
        # --------------------------------------------------------------
        if self._session.state == State.WAITING_FOR_PROMPT:
            prompt = self._extract_user_input(body)
            self._session.original_prompt = prompt
            self._session.current_prompt = prompt

            ambiguities = self._identify_ambiguities(prompt)
            if ambiguities:
                questions = self._generate_clarifying_questions(ambiguities)
                self._session.questions_list = questions
                self._session.state = State.WAITING_FOR_ANSWERS
                return self._make_question_message(prompt, questions)
            else:
                final = self._improve_prompt(prompt)
                self._session = SessionState()          # reset
                return f"✅ Final Prompt (no ambiguities):\n{final}"

        # --------------------------------------------------------------
        # 2️⃣ Follow‑up handling – user answered the clarifying questions
        # --------------------------------------------------------------
        raw_content = _extract_user_input(body)
        if not raw_content:
            return "❗ No answer content received."

        # Split on double‑newlines, ignore empty chunks
        chunks = [c.strip() for c in raw_content.split("\n\n") if c.strip()]
        try:
            answers_dict = {
                int(c.split(".", 1)[0]): c.split(".", 1)[1].strip()
                for c in chunks
            }
        except (ValueError, IndexError) as exc:
            log.exception("Failed to parse numbered answers")
            return f"❗ Unable to parse your answers: {exc}"

        # Preserve the original order of questions
        ordered_answers = [
            answers_dict[i]
            for i in range(1, len(self._session.questions_list) + 1)
            if i in answers_dict
        ]

        if not ordered_answers:
            return "❗ No valid answers extracted from your response."

        new_prompt = self._apply_feedback(ordered_answers)
        self._session = SessionState()                  # reset for next run
        return f"✅ Synthesised Prompt:\n{new_prompt}"

    # ------------------------------------------------------------------
    # Identify ambiguities (calls the LLM – same logic as before)
    # ------------------------------------------------------------------
    def _identify_ambiguities(self, prompt: str) -> List[Dict]:
        analysis = f"""
Identify all potential ambiguities in the following prompt. Return a JSON list.
Each item must contain:
  "ambiguity": <the ambiguous element>
  "description": <why it is ambiguous>

Prompt: {prompt}
"""
        payload = {
            "model": self.valves.PROMPT_IMPROVER_MODEL,
            "messages": [{"role": "user", "content": analysis}],
            "stream": False,
        }
        try:
            resp = _post_json(self.valves.OLLAMA_BASE_URL, payload)
            content = resp["choices"][0]["message"]["content"].strip()
            parsed = json.loads(content)
            return parsed if isinstance(parsed, list) else []
        except Exception as exc:                     # broad on purpose – fallback
            log.exception("Ambiguity detection failed")
            return [{
                "ambiguity": "general",
                "description": "Prompt contains multiple possible interpretations"
            }]

    # ------------------------------------------------------------------
    # Generate clarifying questions from the ambiguities
    # ------------------------------------------------------------------
    def _generate_clarifying_questions(self, ambiguities: List[Dict]) -> List[Dict]:
        prompt = (
            f'For the ambiguities in "{self._session.original_prompt}" '
            "generate ONE clear clarifying question per ambiguity.\n"
        )
        for amb in ambiguities:
            prompt += f"- {json.dumps(amb)}\n"
        prompt += "\nReturn ONLY a JSON list of objects with keys 'question' and 'ambiguity'."

        payload = {
            "model": self.valves.PROMPT_IMPROVER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        try:
            resp = _post_json(self.valves.OLLAMA_BASE_URL, payload)
            content = resp["choices"][0]["message"]["content"].strip()
            parsed = json.loads(content)
            if isinstance(parsed, list):
                clean = [
                    {"question": e["question"].strip(),
                     "ambiguity": e["ambiguity"].strip()}
                    for e in parsed
                    if isinstance(e, dict) and "question" in e and "ambiguity" in e
                ]
                return clean or [{"question": "Please clarify.", "ambiguity": "unknown"}]
        except Exception as exc:
            log.exception("Question generation failed")

        # Fallback – generic question per ambiguity
        return [
            {"question": f"Please clarify: {a.get('ambiguity', 'unknown')}",
             "ambiguity": a.get("ambiguity", "unknown")}
            for a in ambiguities
        ]

    # ------------------------------------------------------------------
    # Format the clarifying‑question message shown to the user
    # ------------------------------------------------------------------
    @staticmethod
    def _make_question_message(original_prompt: str,
                               questions: List[Dict]) -> str:
        table = "## Clarifying Questions\n\n"
        table += "| Index | Ambiguity | Clarifying Question |\n"
        table += "|-------|-----------|----------------------|\n"
        for i, q in enumerate(questions, 1):
            table += f"| {i} | {q['ambiguity']} | {q['question']} |\n"

        return f"""Your prompt has been analysed for potential ambiguities.
Please answer the following questions to help improve it:

Original Prompt:
{original_prompt}

{table}
Respond with your answers in the format:
1. [Answer to first question]
2. [Answer to second question]
..."""

    # ------------------------------------------------------------------
    # Apply the collected answers and ask the LLM for a refined prompt
    # ------------------------------------------------------------------
    def _apply_feedback(self, answers: List[str]) -> str:
        if not answers or not self._session.questions_list:
            return self._session.current_prompt

        qa_pairs = [
            (q["question"].strip(), a.strip())
            for q, a in zip(self._session.questions_list, answers)
        ]

        synthesis = (
            f"Generate a new prompt based on the current prompt:\n"
            f"{self._session.current_prompt}\n"
            f"but with the following questions answered:\n"
        )
        for q, a in qa_pairs:
            synthesis += f"- question: {q}\n  answer: {a}\n"
        synthesis += "\nReturn **only** the fully‑formed new prompt."

        payload = {
            "model": self.valves.PROMPT_IMPROVER_MODEL,
            "messages": [{"role": "user", "content": synthesis}],
            "stream": False,
        }
        try:
            resp = _post_json(self.valves.OLLAMA_BASE_URL, payload)
            new_prompt = resp["choices"][0]["message"]["content"].strip()
            return new_prompt or self._session.current_prompt
        except Exception as exc:
            log.exception("Synthesis failed")
            return f"Exception during synthesis: {exc}"

    # ------------------------------------------------------------------
    # Simple improvement when there are no ambiguities
    # ------------------------------------------------------------------
    def _improve_prompt(self, prompt: str) -> str:
        prompt_req = f"""
Improve the following prompt to make it more specific and easier for an LLM to answer in detail:

Original Prompt: {prompt}
Return only the improved prompt.
"""
        payload = {
            "model": self.valves.PROMPT_IMPROVER_MODEL,
            "messages": [{"role": "user", "content": prompt_req}],
            "stream": False,
        }
        try:
            resp = _post_json(self.valves.OLLAMA_BASE_URL, payload)
            improved = resp["choices"][0]["message"]["content"].strip()
            return improved or prompt
        except Exception as exc:
            log.exception("Improvement failed")
            return prompt


# ----------------------------------------------------------------------
# CLI entry point – just forwards the JSON payload from stdin to the pipe
# ----------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python prompt_improver_stdlib.py <path-to-request-json>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        request_body = json.load(f)

    pipe = PromptImprover()
    response = pipe.pipe(request_body)
    print(response)


if __name__ == "__main__":
    main()