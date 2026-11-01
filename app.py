"""
롯데홈쇼핑 프로모션 시뮬레이터 — 데모 앱
실행: streamlit run app.py
"""
import json
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data_gen import CARD_DISC_LEVELS, GIFT_LEVELS, INSTALLMENT_LEVELS  # noqa: E402
from report import insights, make_report  # noqa: E402
from simulator import ASSUMPTIONS, WEEKDAYS, Simulator  # noqa: E402

st.set_page_config(page_title="프로모션 시뮬레이터", page_icon="🛒", layout="wide")


@st.cache_resource
def get_sim():
    return Simulator()


sim = get_sim()
products = sim.products


def won(x):
    return f"{x / 1e6:,.1f}백만"


def label(pid):
    p = sim.product(pid)
    return f"{p.product_name} ({p.category}, {p.price:,.0f}원)"


# ── 사이드바: 방송 조건 ─────────────────────────────────────
with st.sidebar:
    st.header("📺 방송 조건")
    cat = st.selectbox("카테고리", ["전체"] + sorted(products.category.unique()))
    pool = products if cat == "전체" else products[products.category == cat]
    pid = st.selectbox("상품", pool.product_id, format_func=label)
    weekday = st.selectbox("요일", range(7), index=1, format_func=lambda i: WEEKDAYS[i] + "요일")
    hour = st.slider("방송 시각", 6, 25, 21, help="25 = 새벽 1시")
    card_share = st.slider("카드사 분담률", 0.0, 1.0, 0.5, 0.1, help="카드 청구할인 비용 중 카드사가 부담하는 비율")
    gift_share = st.slider("사은품 협력사 분담률", 0.0, 1.0, 0.0, 0.1, help="사은품 원가 중 협력사(납품업체)가 부담하는 비율")
    recent = st.number_input("최근 2주 동일 상품 프로모션 횟수", 0, 10, 0, help="많을수록 프로모션 피로도↑")
    st.caption("⚠️ 데모는 합성(가상) 데이터로 학습했습니다.")

st.title("🛒 홈쇼핑 프로모션 시뮬레이터 + AI 에이전트")
st.caption("카드할인·무이자·사은품 조건별 주문·매출·**공헌이익**을 미리 시뮬레이션하고 비교합니다.")

tab1, tab2, tab3, tab4 = st.tabs(["📊 시나리오 비교", "🔍 최적 조합 찾기", "🤖 AI 에이전트", "✅ 모델 신뢰도"])

# ── 탭1: 시나리오 비교 ───────────────────────────────────────
with tab1:
    defaults = [dict(card_disc=10, installment=0, gift_pct=0), dict(card_disc=0, installment=12, gift_pct=0),
                dict(card_disc=0, installment=0, gift_pct=8)]
    cols = st.columns(3)
    scenarios = []
    for i, (c, d) in enumerate(zip(cols, defaults)):
        with c:
            st.markdown(f"**시나리오 {'ABC'[i]}**")
            s = dict(
                card_disc=st.select_slider("카드 청구할인(%)", CARD_DISC_LEVELS, d["card_disc"], key=f"c{i}"),
                installment=st.select_slider("무이자(개월)", INSTALLMENT_LEVELS, d["installment"], key=f"i{i}"),
                gift_pct=st.select_slider("사은품(판매가 대비 %)", GIFT_LEVELS, d["gift_pct"], key=f"g{i}"),
                card_share=card_share, gift_vendor_share=gift_share, recent_promo_cnt=recent,
            )
            s["name"] = f"{'ABC'[i]}: {sim.describe(s)}"
            scenarios.append(s)

    res = sim.simulate(pid, weekday, hour, scenarios)
    best = res.iloc[1:].loc[res.iloc[1:].공헌이익.idxmax()]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("이익 기준 추천", best.scenario.split(": ")[0], best.scenario.split(": ")[-1])
    m2.metric("예상 주문", f"{best.예상주문:,.0f}건", f"{best.주문_하한:,.0f}~{best.주문_상한:,.0f}", delta_color="off")
    m3.metric("순매출", won(best.순매출), f"{best.매출증감률:+.0%}")
    m4.metric("공헌이익", won(best.공헌이익), won(best.이익증감))

    fig = go.Figure()
    fig.add_bar(name="순매출", x=res.scenario, y=res.순매출 / 1e6, marker_color="#B0B7C3")
    fig.add_bar(name="공헌이익", x=res.scenario, y=res.공헌이익 / 1e6, marker_color="#E60012",
                error_y=dict(type="data", symmetric=False, array=(res.이익_상한 - res.공헌이익) / 1e6,
                             arrayminus=(res.공헌이익 - res.이익_하한) / 1e6))
    fig.update_layout(barmode="group", yaxis_title="백만원", height=380, margin=dict(t=20, b=20),
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("오차막대 = 80% 예측 범위")

    st.subheader("핵심 메시지")
    for msg in insights(res):
        st.markdown(f"- {msg}")

    with st.expander("상세 표 보기"):
        st.dataframe(res.drop(columns=["card_share", "gift_vendor_share"]), use_container_width=True, hide_index=True)
    st.download_button("📄 회의용 리포트 다운로드(.md)",
                       make_report(sim.product(pid), weekday, hour, res),
                       file_name=f"promo_report_{pid}.md")

# ── 탭2: 최적 조합 ───────────────────────────────────────────
with tab2:
    c1, c2 = st.columns(2)
    objective = c1.radio("목표", ["profit", "sales"], horizontal=True,
                         format_func=lambda x: "공헌이익 최대" if x == "profit" else "매출 최대")
    cap = c2.slider("프로모션비용 상한 (순매출 대비 %)", 0, 20, 20) / 100
    opt = sim.optimize(pid, weekday, hour, objective, card_share, gift_share, None if cap >= 0.2 else cap,
                       top_n=10, recent_promo_cnt=recent)
    st.caption("카드할인 5단계 × 무이자 5단계 × 사은품 5단계 = 125개 조합 전체를 시뮬레이션")
    show = opt[["scenario", "예상주문", "순매출", "프로모션비용", "공헌이익", "매출증감률", "이익증감", "반품률"]].copy()
    st.dataframe(show.style.format({"예상주문": "{:,.0f}", "순매출": "{:,.0f}", "프로모션비용": "{:,.0f}",
                                    "공헌이익": "{:,.0f}", "매출증감률": "{:+.0%}", "이익증감": "{:+,.0f}",
                                    "반품률": "{:.1%}"}), use_container_width=True, hide_index=True)

# ── 탭3: AI 에이전트 ─────────────────────────────────────────
with tab3:
    key = os.getenv("OPENAI_API_KEY") or st.text_input("OpenAI API Key", type="password")
    st.caption("예: '화요일 21시 무선청소기, 카드 10%랑 무이자 12개월 중 뭐가 나아?' / "
               "'토요일 오전 10시 한우세트 이익 최대 조합 찾아줘, 프로모션비는 매출의 8% 이내'")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for m in st.session_state.chat:
        st.chat_message(m["role"]).markdown(m["content"])
    q = st.chat_input("프로모션 기획 질문을 입력하세요")
    if q:
        if not key:
            st.warning("OpenAI API Key가 필요합니다.")
        else:
            os.environ["OPENAI_API_KEY"] = key
            from agent import PromoAgent
            st.chat_message("user").markdown(q)
            agent = PromoAgent(sim=sim)
            hist = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat[-6:]]
            with st.spinner("시뮬레이션 중..."):
                ans = agent.ask(q, history=hist)
            st.chat_message("assistant").markdown(ans)
            with st.expander(f"🔧 에이전트가 호출한 도구 {len(agent.trace)}건"):
                for t in agent.trace:
                    st.code(f"{t['tool']}({json.dumps(t['args'], ensure_ascii=False)})")
            st.session_state.chat += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]

# ── 탭4: 모델 신뢰도 ─────────────────────────────────────────
with tab4:
    mt = json.load(open(os.path.join(os.path.dirname(__file__), "models/metrics.json"), encoding="utf-8"))
    up = pd.read_csv(os.path.join(os.path.dirname(__file__), "models/uplift_check.csv"))
    c1, c2, c3 = st.columns(3)
    c1.metric("미래 기간 주문 예측 오차(WAPE)", f"{mt['test_wape']:.1%}")
    c2.metric("80% 예측범위 실제 적중률", f"{mt['interval_coverage_80']:.1%}")
    c3.metric("프로모션 효과 추정 오차(평균)", f"{mt['uplift_mae'] * 100:.1f}%p")
    st.markdown(f"학습 {mt['train_rows']:,}건({mt['calib_date']} 이전) · 범위 보정 {mt['calib_rows']:,}건 · "
                f"검증 {mt['test_rows']:,}건({mt['split_date']} 이후) — **학습에 쓰지 않은 미래 데이터로 평가**")
    st.subheader("모델이 찾은 프로모션 효과 vs 실제 정답")
    up["label"] = up.lever + " · " + up.category
    fig = go.Figure()
    fig.add_bar(name="정답(합성 데이터 생성 규칙)", x=up.label, y=up.true_uplift * 100, marker_color="#B0B7C3")
    fig.add_bar(name="모델 추정", x=up.label, y=up.model_uplift * 100, marker_color="#E60012")
    fig.update_layout(barmode="group", yaxis_title="주문 증가율(%)", height=400, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"단순 평균 비교의 함정: 가전 카드할인 방송은 할인 안 한 방송보다 주문이 "
            f"{mt['naive_appliance_card_uplift']:.0%} 많지만, 이는 할인이 황금시간대·인기상품에 몰린 효과가 섞인 값입니다. "
            "모델은 시간대·상품을 함께 고려해 프로모션 자체의 효과를 분리합니다.")
    with st.expander("계산 가정"):
        st.json(ASSUMPTIONS)
