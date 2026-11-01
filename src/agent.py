"""
프로모션 기획 AI 에이전트 (OpenAI Responses API + function calling).

MD/방송기획자가 자연어로 물으면, 에이전트가 시뮬레이터를 '도구'로 호출해
숫자에 근거한 비교·추천을 돌려준다. LLM은 계산을 직접 하지 않고, 모든 숫자는 도구 결과에서만 가져온다.

  사용자: "다음 주 화요일 21시 무선청소기, 카드 10%랑 무이자 12개월 중 뭐가 나아?"
  에이전트: search_products → simulate_promotions → 결과 해석 → 추천 + 근거 + 리스크
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

from simulator import ASSUMPTIONS, Simulator

MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

SYSTEM_PROMPT = f"""너는 롯데홈쇼핑 방송 프로모션 기획을 돕는 AI 에이전트다.
역할: MD·방송기획자의 질문을 받아 시뮬레이터 도구로 프로모션 조건별 주문·매출·공헌이익을 계산하고 비교·추천한다.

규칙
1. 숫자는 반드시 도구 결과에서만 인용한다. 직접 추정하거나 지어내지 않는다.
2. 상품이 모호하면 search_products로 찾는다. 요일·시간이 없으면 사용자에게 묻지 말고 가정한 값을 명시한다(기본: 화요일 21시).
3. 판단 기준은 '매출'만이 아니라 '공헌이익'이다. 매출은 오르지만 이익이 줄어드는 조건은 반드시 경고한다.
4. 예측에는 불확실성이 있으므로 예상 범위(하한~상한)를 함께 말한다.
5. 답변 형식: ① 결론 한 줄 ② 비교표(마크다운) ③ 근거 ④ 리스크·주의사항 ⑤ 다음 액션.
6. 카드사 분담률이 주어지지 않으면 50%, 사은품 협력사 분담률은 0%로 가정하고 명시한다.

계산 가정: {json.dumps(ASSUMPTIONS, ensure_ascii=False)}
요일은 '월'~'일', 시간은 6~25(25=새벽 1시) 범위다. 프로모션 단계: 카드할인 0/5/7/10/15%, 무이자 0/3/6/10/12개월, 사은품 0/3/5/8/12%(판매가 대비 원가).
"""

_scenario_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "시나리오 이름(선택)"},
        "card_disc": {"type": "number", "description": "카드 청구할인율(%)"},
        "card_share": {"type": "number", "description": "카드사 분담률 0~1 (기본 0.5)"},
        "gift_vendor_share": {"type": "number", "description": "사은품 원가 중 협력사 부담률 0~1 (기본 0)"},
        "installment": {"type": "number", "description": "무이자 할부 개월 수"},
        "gift_pct": {"type": "number", "description": "사은품 원가, 판매가 대비 %"},
        "recent_promo_cnt": {"type": "integer", "description": "최근 2주 동일 상품 프로모션 횟수(피로도)"},
    },
}

TOOLS = [
    {"type": "function", "name": "search_products",
     "description": "상품 목록 검색. 카테고리(가전/뷰티/패션/식품/리빙) 또는 상품명 키워드로 찾는다.",
     "parameters": {"type": "object", "properties": {
         "category": {"type": "string"}, "keyword": {"type": "string"}}}},
    {"type": "function", "name": "get_product_history",
     "description": "상품의 과거 방송 실적 요약(프로모션 유형별 평균 주문, 최고 실적 방송).",
     "parameters": {"type": "object", "properties": {"product_id": {"type": "string"}},
                    "required": ["product_id"]}},
    {"type": "function", "name": "simulate_promotions",
     "description": "같은 방송 조건에서 여러 프로모션 시나리오의 주문·매출·공헌이익을 시뮬레이션하고 '프로모션 없음' 대비 증감을 비교.",
     "parameters": {"type": "object", "properties": {
         "product_id": {"type": "string"},
         "weekday": {"type": "string", "description": "월~일"},
         "hour": {"type": "integer", "description": "방송 시작 시각 6~25"},
         "scenarios": {"type": "array", "items": _scenario_schema}},
         "required": ["product_id", "weekday", "hour", "scenarios"]}},
    {"type": "function", "name": "optimize_promotion",
     "description": "가능한 모든 프로모션 조합(125개)을 시뮬레이션해 목표(이익 또는 매출) 기준 최적 조합 상위 N개를 찾는다.",
     "parameters": {"type": "object", "properties": {
         "product_id": {"type": "string"},
         "weekday": {"type": "string"},
         "hour": {"type": "integer"},
         "objective": {"type": "string", "enum": ["profit", "sales"]},
         "card_share": {"type": "number", "description": "카드사 분담률 0~1"},
         "gift_vendor_share": {"type": "number", "description": "사은품 협력사 부담률 0~1"},
         "max_promo_cost_rate": {"type": "number", "description": "프로모션비용 상한(순매출 대비 비율, 예: 0.08)"},
         "top_n": {"type": "integer"}},
         "required": ["product_id", "weekday", "hour"]}},
]

RESULT_COLS = ["scenario", "예상주문", "주문_하한", "주문_상한", "반품률", "순매출", "프로모션비용",
               "공헌이익", "이익_하한", "이익_상한", "매출증감률", "이익증감", "프로모션ROI"]


def _records(df: pd.DataFrame) -> list[dict]:
    cols = [c for c in RESULT_COLS if c in df.columns]
    return json.loads(df[cols].replace({np.nan: None}).to_json(orient="records", force_ascii=False))


class PromoAgent:
    def __init__(self, sim: Simulator | None = None, client: Any = None, model: str = MODEL):
        self.sim = sim or Simulator()
        self.model = model
        if client is None:
            from openai import OpenAI
            client = OpenAI()
        self.client = client
        self.trace: list[dict] = []   # 어떤 도구를 어떤 인자로 불렀는지 기록 (설명가능성)

    # ── 도구 실행 ───────────────────────────────────────────
    def call_tool(self, name: str, args: dict) -> Any:
        s = self.sim
        if name == "search_products":
            return s.list_products(args.get("category"), args.get("keyword")).to_dict("records")
        if name == "get_product_history":
            return s.product_history(args["product_id"])
        if name == "simulate_promotions":
            return _records(s.simulate(args["product_id"], args["weekday"], args["hour"], args["scenarios"]))
        if name == "optimize_promotion":
            return _records(s.optimize(args["product_id"], args["weekday"], args["hour"],
                                       objective=args.get("objective", "profit"),
                                       card_share=args.get("card_share", 0.5),
                                       gift_vendor_share=args.get("gift_vendor_share", 0.0),
                                       max_promo_cost_rate=args.get("max_promo_cost_rate"),
                                       top_n=args.get("top_n", 5)))
        raise ValueError(f"알 수 없는 도구: {name}")

    # ── 대화 ──────────────────────────────────────────────
    def ask(self, question: str, history: list[dict] | None = None, max_steps: int = 8) -> str:
        items: list[Any] = list(history or []) + [{"role": "user", "content": question}]
        for _ in range(max_steps):
            resp = self.client.responses.create(model=self.model, instructions=SYSTEM_PROMPT,
                                                input=items, tools=TOOLS)
            calls = [o for o in resp.output if getattr(o, "type", None) == "function_call"]
            if not calls:
                return resp.output_text
            items += resp.output
            for c in calls:
                args = json.loads(c.arguments or "{}")
                try:
                    result = self.call_tool(c.name, args)
                except Exception as e:  # 도구 오류도 모델에게 알려 스스로 고치게 한다
                    result = {"error": str(e)}
                self.trace.append({"tool": c.name, "args": args, "result": result})
                items.append({"type": "function_call_output", "call_id": c.call_id,
                              "output": json.dumps(result, ensure_ascii=False, default=str)})
        return "도구 호출 횟수 한도에 도달했습니다. 질문을 더 구체적으로 해 주세요."


if __name__ == "__main__":
    import sys
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY 환경변수를 설정하세요.")
    q = " ".join(sys.argv[1:]) or "화요일 21시 무선청소기 방송, 카드 10% 할인과 무이자 12개월 중 뭐가 나아?"
    agent = PromoAgent()
    print(agent.ask(q))
    print("\n[도구 호출 기록]")
    for t in agent.trace:
        print("-", t["tool"], t["args"])
