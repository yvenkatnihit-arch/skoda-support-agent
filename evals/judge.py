"""LLM-as-judge for faithfulness and refusals: a separate Gemini call (temperature 0) that only grades."""

import json

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graph.nodes.context import clip
from app.llm import STRUCTURED_OUTPUT_METHOD, LoggedLLM, thinking_kwargs

EVIDENCE_MAX_CHARS = 15000
JUDGE_THINKING_BUDGET = 512

JUDGE_PROMPT = """\
You grade a Škoda India customer-support assistant. You get the customer's question, the assistant's reply,
and the EVIDENCE: the raw results of the tools the assistant used (prices, dealers, search passages...).

1. List the factual claims in the reply about cars, prices, specs, dealers, services or policies.
   Skip greetings, offers to help, questions to the customer, and generic advice such as "confirm with your dealer".
2. For each claim decide if it is SUPPORTED: stated in the evidence or directly implied by it, without
   new arithmetic. A claim with no evidence behind it is NOT supported, even if it is true in the real world.
3. declined: true if the reply says it can't help with the request, that it doesn't have the information,
   or that something isn't available; false if it answers the request.
Be strict and brief. Reasons: one short sentence each."""


class Claim(BaseModel):
    claim: str
    supported: bool
    reason: str = ""


class Verdict(BaseModel):
    claims: list[Claim] = Field(default_factory=list)
    declined: bool = Field(description="The reply declines or says the information isn't available")


def make_judge():
    from langchain_google_genai import ChatGoogleGenerativeAI

    s = get_settings()
    model = s.default_model
    llm = ChatGoogleGenerativeAI(model=model, google_api_key=s.google_api_key, timeout=s.llm_timeout_s,
                                 max_retries=1, temperature=0, max_output_tokens=4096 + JUDGE_THINKING_BUDGET,
                                 **thinking_kwargs(model, JUDGE_THINKING_BUDGET))
    structured = llm.with_structured_output(Verdict, method=STRUCTURED_OUTPUT_METHOD, include_raw=True)
    return LoggedLLM(structured, "eval_judge", model, s)


def judge(judge_llm, question: str, reply: str, agent_outputs: dict) -> Verdict:
    evidence = [{"tool": r["tool"], "args": r.get("args"), "result": r.get("content")}
                for out in (agent_outputs or {}).values() for r in out.get("tool_results", [])]
    payload = (f"QUESTION:\n{question}\n\nREPLY:\n{reply}\n\n"
               f"EVIDENCE:\n{clip(json.dumps(evidence, ensure_ascii=False), EVIDENCE_MAX_CHARS) or '(none)'}")
    return judge_llm.invoke([SystemMessage(JUDGE_PROMPT), HumanMessage(payload)])


def faithfulness(verdict: Verdict) -> float:
    """Share of claims supported by the evidence; 1.0 when the reply makes no factual claims."""
    if not verdict.claims:
        return 1.0
    return sum(c.supported for c in verdict.claims) / len(verdict.claims)
