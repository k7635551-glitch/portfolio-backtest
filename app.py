"""
포트폴리오 백테스트 웹앱 (Streamlit) — 개선판
- 단계별 설명 UI
- 차트 + 텍스트 결과 병행 표시
- Claude API 해석 리포트 자동 생성
"""

import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
import json
import requests
from datetime import date

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# ════════════════════════════════════════════════════════════
#  페이지 설정
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="포트폴리오 백테스터",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

[data-testid="metric-container"] {
    background: #1a1d2e;
    border: 1px solid #2e3250;
    border-radius: 10px;
    padding: 12px 16px;
}
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #00d4aa, #0098cc);
    color: white;
    font-weight: bold;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 1rem;
}
.stButton > button:hover { opacity: 0.88; }

.guide-box {
    background: #1a1d2e;
    border-left: 4px solid #00d4aa;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0 16px 0;
    font-size: 0.9rem;
    color: #c8cad8;
    line-height: 1.7;
}
.section-title {
    font-size: 1.1rem;
    font-weight: bold;
    color: #00d4aa;
    margin: 20px 0 6px 0;
}
.result-card {
    background: #1a1d2e;
    border: 1px solid #2e3250;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
@media (max-width: 768px) {
    h1 { font-size: 1.4rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  기본 설정
# ════════════════════════════════════════════════════════════
DEFAULT_TICKERS = {
    "금현물":    "IAU",
    "나스닥100":  "QQQ",
    "현금(소파)": "UUP",
    "한국리츠":   "088980.KS",
}
DEFAULT_IS_USD = {
    "금현물": True, "나스닥100": True, "현금(소파)": True, "한국리츠": False,
}
PALETTE = ["#00d4aa", "#4fc3f7", "#ffb74d", "#ff6b6b"]
BG, PANEL, TEXT, GRID = "#0f1117", "#1a1d2e", "#e8eaf0", "#2e3250"


# ════════════════════════════════════════════════════════════
#  데이터 수집
# ════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(tickers, is_usd, start, end):
    symbols = list(tickers.values())
    raw = yf.download(symbols, start=start, end=end, auto_adjust=False, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=symbols[0])
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    inv_map = {v: k for k, v in tickers.items()}
    raw.rename(columns=inv_map, inplace=True)
    raw.dropna(how="all", inplace=True)
    raw.ffill(inplace=True)

    fx_raw = yf.download("KRW=X", start=start, end=end, auto_adjust=False, progress=False)["Close"]
    if isinstance(fx_raw, pd.DataFrame):
        fx_raw = fx_raw.squeeze()
    fx_raw = fx_raw.reindex(raw.index).ffill().bfill()

    for name in tickers.keys():
        if is_usd.get(name, False):
            raw[name] = raw[name] * fx_raw

    return raw, fx_raw


# ════════════════════════════════════════════════════════════
#  입금 스케줄
# ════════════════════════════════════════════════════════════
def build_deposit_schedule(cfg, trade_index):
    deposits = {}
    if cfg.get("initial", 0) > 0:
        first_day = trade_index[0]
        deposits[first_day] = deposits.get(first_day, 0) + cfg["initial"]
    if cfg.get("dca_amount", 0) > 0 and cfg.get("dca_freq"):
        sched = pd.date_range(trade_index[0], trade_index[-1], freq=cfg["dca_freq"])
        for d in sched:
            future = trade_index[trade_index >= d]
            if len(future) == 0:
                continue
            tday = future[0]
            deposits[tday] = deposits.get(tday, 0) + cfg["dca_amount"]
    for date_str, amount in cfg.get("custom_deposits", {}).items():
        d = pd.Timestamp(date_str)
        future = trade_index[trade_index >= d]
        if len(future) == 0:
            continue
        tday = future[0]
        deposits[tday] = deposits.get(tday, 0) + amount
    return pd.Series(deposits).sort_index() if deposits else pd.Series(dtype=float)


# ════════════════════════════════════════════════════════════
#  백테스트 엔진
# ════════════════════════════════════════════════════════════
def run_backtest(prices, weights, monthly_yields, cfg, cost):
    names = list(weights.keys())
    w      = np.array([weights[n] for n in names])
    my_arr = np.array([monthly_yields[n] for n in names])
    deposit_sched = build_deposit_schedule(cfg, prices.index)

    rebal_dates = set()
    for year in prices.index.year.unique():
        for month in [1, 7]:
            mask = (prices.index.year == year) & (prices.index.month == month)
            if mask.any():
                rebal_dates.add(prices.index[mask][0])

    shares = np.zeros(len(names))
    cash = 0.0
    total_invested = 0.0
    portfolio_val, invested_val = [], []
    prev_month = None

    for date_idx, row in prices.iterrows():
        price_arr  = row[names].values.astype(float)
        safe_price = np.where(price_arr > 0, price_arr, 1.0)

        if date_idx.month != prev_month:
            dividend_gain = (shares * price_arr) * my_arr
            shares += dividend_gain / safe_price
            prev_month = date_idx.month

        if len(deposit_sched) > 0 and date_idx in deposit_sched.index:
            inflow = deposit_sched[date_idx]
            cash += inflow
            total_invested += inflow
            shares += (cash * w) / safe_price
            cash = 0.0

        if date_idx in rebal_dates and shares.sum() > 0:
            current_val   = (shares * price_arr).sum()
            current_alloc = shares * price_arr
            target_alloc  = current_val * w
            trade_amounts = np.abs(target_alloc - current_alloc)
            total_cost    = (trade_amounts * cost).sum()
            net_val       = current_val - total_cost
            shares        = (net_val * w) / safe_price

        val = (shares * price_arr).sum() + cash
        portfolio_val.append(val)
        invested_val.append(total_invested)

    return (pd.Series(portfolio_val, index=prices.index),
            pd.Series(invested_val,  index=prices.index))


# ════════════════════════════════════════════════════════════
#  성과 지표
# ════════════════════════════════════════════════════════════
def calc_metrics(pf, inv):
    total_invested = inv.iloc[-1]
    final_val      = pf.iloc[-1]
    total_ret      = (final_val - total_invested) / total_invested if total_invested > 0 else 0

    inv_diff = inv.diff().fillna(0)
    inv_diff.iloc[0] = inv.iloc[0]
    cash_flows = inv_diff[inv_diff > 0] * -1.0
    if pf.index[-1] in cash_flows.index:
        cash_flows[pf.index[-1]] += final_val
    else:
        cash_flows[pf.index[-1]] = final_val
    full_cf = pd.Series(0.0, index=pf.index)
    for d, val in cash_flows.sort_index().items():
        full_cf[d] += val

    daily_irr = npf.irr(full_cf.values)
    cagr = (1 + daily_irr) ** 252 - 1 if not np.isnan(daily_irr) else 0

    inv_diff2 = inv.diff().fillna(0)
    inv_diff2.iloc[0] = inv.iloc[0]
    deposit_days = set(inv_diff2[inv_diff2 > 0].index)
    daily_chg = pf.diff().fillna(0)
    prev_val  = pf.shift(1).bfill()
    twr = pd.Series(0.0, index=pf.index)
    for d in pf.index:
        if d in deposit_days or prev_val[d] <= 0:
            continue
        twr[d] = daily_chg[d] / prev_val[d]

    vol    = twr.std() * np.sqrt(252)
    sharpe = (cagr - 0.035) / vol if vol > 0 else 0
    dd     = (pf - pf.cummax()) / pf.cummax()
    mdd    = dd.min()
    years  = (pf.index[-1] - pf.index[0]).days / 365.25

    return {
        "total_invested": total_invested,
        "final_val":      final_val,
        "profit":         final_val - total_invested,
        "total_ret":      total_ret * 100,
        "cagr":           cagr * 100,
        "vol":            vol * 100,
        "sharpe":         sharpe,
        "mdd":            mdd * 100,
        "years":          years,
    }


# ════════════════════════════════════════════════════════════
#  차트
# ════════════════════════════════════════════════════════════
def make_chart(results):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), facecolor=BG)
    fig.subplots_adjust(hspace=0.45)

    def style_ax(ax):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.grid(alpha=0.15, color=GRID)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)

    ax1, ax2 = axes
    style_ax(ax1)
    style_ax(ax2)

    for i, (name, (pf, inv)) in enumerate(results.items()):
        c = PALETTE[i % len(PALETTE)]
        ax1.plot(pf.index, pf / 1e6, color=c, lw=1.8, label=f"{name}")
        ax1.plot(inv.index, inv / 1e6, color=c, lw=0.9, ls=":", alpha=0.6)

    ax1.set_title("자산가치 추이  (실선=자산, 점선=투입금)", fontsize=10,
                  fontweight="bold", color=TEXT)
    ax1.set_ylabel("금액 (백만원)", color=TEXT)
    ax1.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8,
               ncol=len(results), loc="upper left")

    for i, (name, (pf, inv)) in enumerate(results.items()):
        c  = PALETTE[i % len(PALETTE)]
        dd = (pf - pf.cummax()) / pf.cummax() * 100
        ax2.fill_between(dd.index, dd, 0, alpha=0.2, color=c)
        ax2.plot(dd.index, dd, color=c, lw=1.1, label=name)

    ax2.set_title("낙폭(Drawdown) 비교", fontsize=10, fontweight="bold", color=TEXT)
    ax2.set_ylabel("낙폭 (%)", color=TEXT)
    ax2.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8, ncol=len(results))

    return fig


# ════════════════════════════════════════════════════════════
#  텍스트 결과표
# ════════════════════════════════════════════════════════════
def show_text_results(results, weights, monthly_yields, cost, start_date, end_date):
    st.markdown("### 📋 시나리오별 성과 요약")

    rows = []
    for name, (pf, inv) in results.items():
        m = calc_metrics(pf, inv)
        rows.append({
            "시나리오":      name,
            "총 투입금":     f"₩{m['total_invested']/1e8:.2f}억",
            "최종 자산":     f"₩{m['final_val']/1e8:.2f}억",
            "수익금":        f"₩{m['profit']/1e6:+,.0f}만",
            "누적 수익률":   f"{m['total_ret']:+.2f}%",
            "CAGR":          f"{m['cagr']:+.2f}%",
            "연간 변동성":   f"{m['vol']:.2f}%",
            "샤프 비율":     f"{m['sharpe']:.2f}",
            "최대 낙폭":     f"{m['mdd']:.2f}%",
        })

    df = pd.DataFrame(rows).set_index("시나리오")
    st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📌 시나리오별 상세 지표")
    for i, (name, (pf, inv)) in enumerate(results.items()):
        m = calc_metrics(pf, inv)
        color = PALETTE[i % len(PALETTE)]
        st.markdown(
            f"<div class='section-title' style='color:{color}'>▶ {name}</div>",
            unsafe_allow_html=True
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("총 투입금",   f"₩{m['total_invested']/1e8:.2f}억")
        c2.metric("최종 자산",   f"₩{m['final_val']/1e8:.2f}억")
        c3.metric("수익금",      f"₩{m['profit']/1e6:+,.0f}만",
                  delta=f"{m['total_ret']:+.1f}%")
        c4.metric("CAGR",        f"{m['cagr']:+.2f}%")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("연간 변동성", f"{m['vol']:.2f}%")
        c6.metric("샤프 비율",   f"{m['sharpe']:.2f}")
        c7.metric("최대 낙폭",   f"{m['mdd']:.2f}%")
        c8.metric("백테스트 기간", f"{m['years']:.1f}년")
        st.markdown("")

    return rows


# ════════════════════════════════════════════════════════════
#  Claude AI 해석 리포트
# ════════════════════════════════════════════════════════════
def generate_report(results, weights, monthly_yields, cost, start_date, end_date):
    metrics_summary = []
    for name, (pf, inv) in results.items():
        m = calc_metrics(pf, inv)
        metrics_summary.append(
            f"[{name}] 투입:{m['total_invested']/1e8:.2f}억 / 최종:{m['final_val']/1e8:.2f}억 / "
            f"수익률:{m['total_ret']:+.1f}% / CAGR:{m['cagr']:+.2f}% / "
            f"변동성:{m['vol']:.1f}% / 샤프:{m['sharpe']:.2f} / MDD:{m['mdd']:.1f}%"
        )

    weight_str = " / ".join([f"{k} {v*100:.0f}%" for k, v in weights.items()])
    yield_str  = " / ".join([f"{k} {v*12*100:.1f}%" for k, v in monthly_yields.items()])

    prompt = f"""당신은 재무 분석 전문가입니다. 아래 포트폴리오 백테스트 결과를 분석하여 투자자가 이해하기 쉬운 한국어 리포트를 작성해주세요.

[백테스트 조건]
- 기간: {start_date} ~ {end_date}
- 포트폴리오 구성: {weight_str}
- 종목: 금현물(IAU), 나스닥100(QQQ), 현금달러소파(UUP), 한국리츠(맥쿼리인프라 088980.KS)
- 달러 자산은 원달러 환율 변동 반영 (원화 기준 수익률)
- 연간 배당/분배금: {yield_str}
- 반기 리밸런싱 (1월·7월), 수수료 {cost*100:.2f}%

[시나리오별 성과]
{chr(10).join(metrics_summary)}

위 결과를 바탕으로 다음 구조로 리포트를 작성해주세요:

1. **전체 요약** (3~4문장으로 핵심만)
2. **시나리오별 분석** (각 시나리오의 특징과 결과 해석)
3. **포트폴리오 특성 분석** (변동성, 샤프비율, MDD 관점)
4. **입금 전략별 적합한 투자자 유형**
5. **주의사항 및 한계점**

전문적이되 일반 투자자도 이해할 수 있는 언어로 작성해주세요. 마크다운 형식으로 작성하세요."""

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"Content-Type": "application/json"},
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    data = response.json()
    return data["content"][0]["text"]


# ════════════════════════════════════════════════════════════
#  메인 UI
# ════════════════════════════════════════════════════════════
st.title("📊 포트폴리오 백테스터")

st.markdown("""
<div class="guide-box">
<b>이 앱은 무엇인가요?</b><br>
금현물·나스닥100·달러소파·한국리츠로 구성된 분산 포트폴리오의 과거 성과를 시뮬레이션합니다.<br>
달러 자산은 실제 원달러 환율 변동을 반영하며, 배당 재투자와 반기 리밸런싱이 자동으로 적용됩니다.<br>
<b>사용 방법:</b> ① 왼쪽 사이드바에서 기본 설정 → ② 시나리오 탭에서 입금 방식 설정 → ③ 백테스트 실행
</div>
""", unsafe_allow_html=True)

# ── 사이드바 ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 기본 설정")

    st.markdown("**📅 백테스트 기간**")
    col_s, col_e = st.columns(2)
    start_date = col_s.date_input("시작일", value=date(2016, 1, 1),
                                  min_value=date(2010, 1, 1), max_value=date(2024, 1, 1))
    end_date   = col_e.date_input("종료일", value=date(2025, 6, 30),
                                  min_value=date(2011, 1, 1), max_value=date(2025, 12, 31))

    st.markdown("---")
    st.markdown("**⚖️ 자산 비중 (%)**")
    st.caption("네 자산의 비중 합계가 100%가 되도록 설정하세요. 100%가 아니면 자동 정규화됩니다.")
    w_gold  = st.slider("🥇 금현물",     0, 100, 25, 5)
    w_nas   = st.slider("💻 나스닥100",  0, 100, 25, 5)
    w_cash  = st.slider("💵 현금(소파)", 0, 100, 25, 5)
    w_reit  = st.slider("🏢 한국리츠",   0, 100, 25, 5)
    total_w = w_gold + w_nas + w_cash + w_reit
    if total_w != 100:
        st.warning(f"비중 합계: {total_w}% → 자동 정규화")
    else:
        st.success("비중 합계: 100% ✅")

    weights_raw = {"금현물": w_gold, "나스닥100": w_nas, "현금(소파)": w_cash, "한국리츠": w_reit}
    weights = {k: v / total_w for k, v in weights_raw.items()}

    st.markdown("---")
    st.markdown("**💰 연간 배당/분배금률 (%)**")
    st.caption("각 자산의 연간 배당 수익률을 입력하세요. 매월 자동 재투자됩니다.")
    y_gold = st.number_input("🥇 금현물 (무배당)",    0.0, 20.0, 0.0, 0.1)
    y_nas  = st.number_input("💻 나스닥100 (ACE ETF)", 0.0, 20.0, 3.0, 0.1)
    y_cash = st.number_input("💵 현금소파 (달러예금)", 0.0, 20.0, 4.0, 0.1)
    y_reit = st.number_input("🏢 한국리츠 (맥쿼리)",  0.0, 20.0, 6.5, 0.1)
    monthly_yields = {
        "금현물": y_gold/100/12, "나스닥100": y_nas/100/12,
        "현금(소파)": y_cash/100/12, "한국리츠": y_reit/100/12,
    }

    st.markdown("---")
    st.markdown("**🔄 리밸런싱 수수료**")
    st.caption("거래금액 기준 수수료율입니다. 1월·7월 첫 거래일에 자동 리밸런싱됩니다.")
    cost = st.slider("수수료 (%)", 0.0, 2.0, 0.5, 0.05) / 100

# ── 시나리오 탭 ──────────────────────────────────────────
st.markdown("### 💼 입금 시나리오 설정")
st.markdown("""
<div class="guide-box">
아래 4가지 탭에서 각 시나리오의 입금 방식을 설정합니다. 여러 시나리오를 동시에 활성화하면 나란히 비교할 수 있습니다.
</div>
""", unsafe_allow_html=True)

tab_a, tab_b, tab_c, tab_d = st.tabs([
    "💰 A: 일시납", "📅 B: 월적립", "🔀 C: 혼합", "🛠️ D: 커스텀"
])

scenarios = {}

with tab_a:
    st.markdown("**💰 일시납** — 전체 자금을 첫날 한 번에 투입하는 방식입니다. 투자 시점의 중요성을 확인할 수 있습니다.")
    en_a  = st.checkbox("이 시나리오 활성화", value=True, key="en_a")
    ini_a = st.number_input("초기 투입금 (원)", 0, 10_000_000_000,
                             100_000_000, 1_000_000, key="ini_a", format="%d")
    st.caption(f"= ₩{ini_a/1e8:.2f}억")
    scenarios["일시납"] = {"enabled": en_a, "initial": ini_a, "dca_amount": 0, "dca_freq": None}

with tab_b:
    st.markdown("**📅 월적립 (DCA)** — 매월 일정 금액을 꾸준히 투자하는 방식입니다. 시장 평균 매입 단가를 낮추는 효과가 있습니다.")
    en_b  = st.checkbox("이 시나리오 활성화", value=True, key="en_b")
    dca_b = st.number_input("월 적립금 (원)", 0, 100_000_000,
                             3_000_000, 500_000, key="dca_b", format="%d")
    st.caption(f"월 ₩{dca_b/1e4:.0f}만 × 12 = 연 ₩{dca_b*12/1e8:.2f}억")
    scenarios["월적립"] = {"enabled": en_b, "initial": 0, "dca_amount": dca_b, "dca_freq": "MS"}

with tab_c:
    st.markdown("**🔀 혼합** — 초기 목돈을 투입한 후 매월 추가 적립하는 방식입니다. 목돈과 적립의 시너지를 확인할 수 있습니다.")
    en_c  = st.checkbox("이 시나리오 활성화", value=True, key="en_c")
    ini_c = st.number_input("초기 투입금 (원)", 0, 10_000_000_000,
                             50_000_000, 1_000_000, key="ini_c", format="%d")
    dca_c = st.number_input("월 추가 적립금 (원)", 0, 100_000_000,
                             2_000_000, 500_000, key="dca_c", format="%d")
    st.caption(f"초기 ₩{ini_c/1e8:.2f}억 + 월 ₩{dca_c/1e4:.0f}만")
    scenarios["혼합"] = {"enabled": en_c, "initial": ini_c, "dca_amount": dca_c, "dca_freq": "MS"}

with tab_d:
    st.markdown("**🛠️ 커스텀** — 특정 시점에 비정기적으로 입금하는 방식입니다. 보너스·연말정산 환급 등 실제 입금 패턴을 재현할 수 있습니다.")
    en_d  = st.checkbox("이 시나리오 활성화", value=True, key="en_d")
    ini_d = st.number_input("초기 투입금 (원)", 0, 10_000_000_000,
                             30_000_000, 1_000_000, key="ini_d", format="%d")
    st.markdown("**비정기 입금 일정** — 한 줄에 하나씩 `날짜: 금액` 형식으로 입력하세요.")
    default_custom = (
        "2022-04-01: 10000000\n"
        "2022-10-01: 15000000\n"
        "2023-03-01: 10000000\n"
        "2023-09-01: 20000000\n"
        "2024-01-01:  5000000"
    )
    custom_text = st.text_area("입금 스케줄", default_custom, height=150, key="custom_text")
    custom_deposits = {}
    for line in custom_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            parts = line.replace(",", "").split(":")
            d_str = parts[0].strip()
            amt   = float(parts[1].strip())
            custom_deposits[d_str] = amt
        except Exception:
            pass
    total_custom = sum(custom_deposits.values())
    st.caption(f"비정기 입금 합계: ₩{total_custom/1e4:.0f}만 ({len(custom_deposits)}건) / 총 예상 투입: ₩{(ini_d+total_custom)/1e8:.2f}억")
    scenarios["커스텀"] = {
        "enabled": en_d, "initial": ini_d, "dca_amount": 0,
        "dca_freq": None, "custom_deposits": custom_deposits,
    }

# ── 실행 버튼 ─────────────────────────────────────────────
st.markdown("---")
run_col, report_col = st.columns([3, 1])
with run_col:
    run_btn = st.button("🚀 백테스트 실행", use_container_width=True)
with report_col:
    report_btn = st.button("🤖 AI 해석 리포트", use_container_width=True,
                            help="백테스트 실행 후 클릭하세요")

# ── 백테스트 실행 ─────────────────────────────────────────
if run_btn or report_btn:
    enabled = {k: v for k, v in scenarios.items() if v.get("enabled", False)}
    if not enabled:
        st.warning("최소 1개 이상의 시나리오를 활성화하세요.")
        st.stop()

    with st.spinner("📡 시장 데이터 수신 중... (최대 30초 소요)"):
        try:
            prices, fx = fetch_prices(
                DEFAULT_TICKERS, DEFAULT_IS_USD,
                str(start_date), str(end_date)
            )
        except Exception as e:
            st.error(f"데이터 수집 실패: {e}")
            st.stop()

    st.markdown(
        f'<div class="guide-box">📈 환율: {str(start_date)[:7]} ≈ <b>{fx.iloc[0]:.0f}원</b> → '
        f'{str(end_date)[:7]} ≈ <b>{fx.iloc[-1]:.0f}원</b> '
        f'(변화: {(fx.iloc[-1]/fx.iloc[0]-1)*100:+.1f}%) — 달러 자산 수익률에 환율 변동이 반영되어 있습니다.</div>',
        unsafe_allow_html=True
    )

    with st.spinner("⚙️ 백테스트 계산 중..."):
        results = {}
        for name, cfg in enabled.items():
            pf, inv = run_backtest(prices, weights, monthly_yields, cfg, cost)
            results[name] = (pf, inv)

    # 차트
    st.markdown("### 📈 시나리오 비교 차트")
    try:
        fig = make_chart(results)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as e:
        st.warning(f"차트 렌더링 오류: {e} — 텍스트 결과를 확인하세요.")

    # 텍스트 결과
    show_text_results(results, weights, monthly_yields, cost, start_date, end_date)

    # JSON 다운로드
    json_out = {}
    for name, (pf, inv) in results.items():
        m     = calc_metrics(pf, inv)
        pf_m  = pf.resample("ME").last()
        inv_m = inv.resample("ME").last()
        dd_m  = ((pf - pf.cummax()) / pf.cummax()).resample("ME").last()
        json_out[name] = {
            "metrics":   {k: round(v, 2) for k, v in m.items()},
            "dates":     [d.strftime("%Y-%m") for d in pf_m.index],
            "portfolio": [round(v) for v in pf_m.values],
            "invested":  [round(v) for v in inv_m.values],
            "drawdown":  [round(v * 100, 2) for v in dd_m.values],
        }
    st.download_button(
        label="⬇️ 결과 JSON 다운로드",
        data=json.dumps(json_out, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="backtest_results.json",
        mime="application/json",
    )

    # AI 해석 리포트
    if report_btn:
        st.markdown("---")
        st.markdown("### 🤖 AI 해석 리포트")
        with st.spinner("리포트 생성 중..."):
            try:
                report = generate_report(results, weights, monthly_yields,
                                         cost, start_date, end_date)
                st.markdown(report)
            except Exception as e:
                st.error(f"리포트 생성 실패: {e}")

# ── 푸터 ─────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "데이터 출처: Yahoo Finance ｜ "
    "달러 자산(IAU·QQQ·UUP) → KRW=X 환율 원화 환산 ｜ "
    "088980.KS(맥쿼리인프라) → 원화 직접 ｜ "
    "본 앱은 투자 참고용이며 실제 수익을 보장하지 않습니다."
)
