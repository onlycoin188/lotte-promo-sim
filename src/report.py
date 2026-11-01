"""방송 프로모션 검토 리포트 자동 생성 (편성·기획 회의용 마크다운)."""
from __future__ import annotations

from datetime import date

import pandas as pd

from simulator import ASSUMPTIONS, WEEKDAYS


def won(x: float) -> str:
    return f"{x / 1e6:,.1f}백만원"


def insights(res: pd.DataFrame) -> list[str]:
    """시뮬레이션 결과에서 규칙 기반 핵심 메시지를 뽑는다 (LLM 없이도 동작)."""
    base = res.iloc[0]
    rest = res.iloc[1:]
    msgs = []
    if rest.empty:
        return msgs
    best = rest.loc[rest.공헌이익.idxmax()]
    top_sales = rest.loc[rest.순매출.idxmax()]
    if best.공헌이익 > base.공헌이익:
        msgs.append(f"이익 기준 최선은 **{best.scenario}** — 프로모션 없음 대비 이익 {won(best.이익증감)} 증가 "
                    f"(매출 {best.매출증감률:+.0%}).")
    else:
        msgs.append("검토한 모든 프로모션이 '프로모션 없음'보다 이익이 낮습니다. 프로모션 없이 방송하거나 조건을 낮추는 것을 검토하세요.")
    if top_sales.scenario != best.scenario:
        msgs.append(f"매출만 보면 **{top_sales.scenario}**가 가장 높지만(매출 {top_sales.매출증감률:+.0%}), "
                    f"이익은 {best.scenario}보다 {won(best.공헌이익 - top_sales.공헌이익)} 적습니다.")
    for _, r in rest.iterrows():
        if r.매출증감률 > 0 and r.이익증감 < 0:
            msgs.append(f"⚠️ **{r.scenario}**: 매출은 {r.매출증감률:+.0%} 오르지만 이익은 {won(-r.이익증감)} 감소합니다.")
        if r.반품률 > base.반품률 * 1.2:
            msgs.append(f"⚠️ **{r.scenario}**: 반품률이 {base.반품률:.1%} → {r.반품률:.1%}로 높아집니다(충동구매).")
    msgs.append(f"💡 개별 방송 실적은 변동이 큽니다. {best.scenario}의 이익은 80% 확률로 "
                f"{won(best.이익_하한)} ~ {won(best.이익_상한)} 범위로 예상됩니다.")
    return msgs


def make_report(product: pd.Series, weekday: int, hour: int, res: pd.DataFrame) -> str:
    cols = ["scenario", "예상주문", "순매출", "프로모션비용", "공헌이익", "매출증감률", "이익증감", "반품률"]
    t = res[cols].copy()
    for c in ["순매출", "프로모션비용", "공헌이익", "이익증감"]:
        t[c] = t[c].map(won)
    t["매출증감률"] = t["매출증감률"].map(lambda x: f"{x:+.0%}")
    t["반품률"] = t["반품률"].map(lambda x: f"{x:.1%}")
    t["예상주문"] = t["예상주문"].map(lambda x: f"{x:,.0f}")
    table = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    table += "\n".join("| " + " | ".join(map(str, row)) + " |" for row in t.itertuples(index=False))
    hh = hour if hour < 24 else hour - 24
    lines = [
        f"# 방송 프로모션 검토 리포트",
        f"- 작성일: {date.today().isoformat()} (자동 생성)",
        f"- 상품: {product.product_name} ({product.category}) / 판매가 {product.price:,.0f}원 / 마진율 {product.margin_rate:.0%}",
        f"- 방송: {WEEKDAYS[weekday]}요일 {hh:02d}시",
        "", "## 핵심 메시지", *[f"- {m}" for m in insights(res)],
        "", "## 시나리오 비교", table,
        "", "## 계산 가정", *[f"- {k}: {v}" for k, v in ASSUMPTIONS.items()],
    ]
    return "\n".join(lines)
