"""
포트폴리오 백테스트 웹앱 (Streamlit)
- 4가지 입금 시나리오 비교
- 달러 자산 원화 환산 (KRW=X)
- 배당/분배금 재투자
- 반기 리밸런싱 (1월·7월)
- 모바일 대응 UI
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
from datetime import date

warnings.filterwarnings("ignore")

# ── 한글 폰트 (서버 환경) ─────────────────────────────────
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

# ── 모바일 대응 CSS ──────────────────────────────────────
st.markdown("""
<style>
/* 전체 폰트 */
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

/* 메트릭 카드 */
[data-testid="metric-container"] {
    background: #1a1d2e;
    border: 1px solid #2e3250;
    border-radius: 10px;
    padding: 12px 16px;
}

/* 모바일: 사이드바 기본 닫힘 처리는 Streamlit이 자동 처리 */
@media (max-width: 768px) {
    [data-testid="metric-container"] { padding: 8px 10px; }
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.1rem !important; }
}

/* 버튼 */
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

/* 탭 */
.stTabs [data-baseweb="tab"] { font-size: 0.9rem; }

/* 경고/정보 박스 */
.info-box {
    background: #1a1d2e;
    border-left: 4px solid #00d4aa;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 0.88rem;
    color: #c8cad8;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  기본 설정값 (수정 가능)
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
DEFAULT_WEIGHTS = {
    "금현물": 0.25, "나스닥100": 0.25, "현금(소파)": 0.25, "한국리츠": 0.25,
}
DEFAULT_YIELDS = {
    "금현물": 0.0, "나스닥100": 0.03/12, "현금(소파)": 0.04/12, "한국리츠": 0.065/12,
}

PALETTE = ["#00d4aa", "#4fc3f7", "#ffb74d", "#ff6b6b", "#ce93d8"]
BG    = "#0f1117"
PANEL = "#1a1d2e"
TEXT  = "#e8eaf0"
GRID  = "#2e3250"

# ════════════════════════════════════════════════════════════
#  데이터 수집
# ════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(tickers: dict, is_usd: dict, start: str, end: str) -> pd.DataFrame:
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
def build_deposit_schedule(cfg: dict, trade_index: pd.DatetimeIndex) -> pd.Series:
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
def run_backtest(prices: pd.DataFrame, weights: dict, monthly_yields: dict,
                 cfg: dict, cost: float):
    names = list(weights.keys())
    w     = np.array([weights[n] for n in names])
    my_arr = np.array([monthly_yields[n] for n in names])

    deposit_sched = build_deposit_schedule(cfg, prices.index)

    rebal_dates = set()
    for year in prices.index.year.unique():
        for month in [1, 7]:
            mask = (prices.index.year == year) & (prices.index.month == month)
            if mask.any():
                rebal_dates.add(prices.index[mask][0])

    shares         = np.zeros(len(names))
    cash           = 0.0
    total_invested = 0.0
    portfolio_val  = []
    invested_val   = []
    prev_month     = None

    for date_idx, row in prices.iterrows():
        price_arr  = row[names].values.astype(float)
        safe_price = np.where(price_arr > 0, price_arr, 1.0)

        if date_idx.month != prev_month:
            dividend_gain = (shares * price_arr) * my_arr
            shares += dividend_gain / safe_price
            prev_month = date_idx.month

        if len(deposit_sched) > 0 and date_idx in deposit_sched.index:
            inflow          = deposit_sched[date_idx]
            cash           += inflow
            total_invested += inflow
            shares         += (cash * w) / safe_price
            cash            = 0.0

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
def calc_metrics(pf: pd.Series, inv: pd.Series) -> dict:
    total_invested = inv.iloc[-1]
    final_val      = pf.iloc[-1]
    total_ret      = (final_val - total_invested) / total_invested if total_invested > 0 else 0

    inv_diff       = inv.diff().fillna(0)
    inv_diff.iloc[0] = inv.iloc[0]
    cash_flows     = inv_diff[inv_diff > 0] * -1.0

    if pf.index[-1] in cash_flows.index:
        cash_flows[pf.index[-1]] += final_val
    else:
        cash_flows[pf.index[-1]] = final_val

    full_cf = pd.Series(0.0, index=pf.index)
    for d, val in cash_flows.sort_index().items():
        full_cf[d] += val

    daily_irr = npf.irr(full_cf.values)
    cagr      = (1 + daily_irr) ** 252 - 1 if not np.isnan(daily_irr) else 0

    inv_diff2       = inv.diff().fillna(0)
    inv_diff2.iloc[0] = inv.iloc[0]
    deposit_days    = set(inv_diff2[inv_diff2 > 0].index)
    daily_chg       = pf.diff().fillna(0)
    prev_val        = pf.shift(1).bfill()
    twr             = pd.Series(0.0, index=pf.index)
    for d in pf.index:
        if d in deposit_days or prev_val[d] <= 0:
            continue
        twr[d] = daily_chg[d] / prev_val[d]

    vol    = twr.std() * np.sqrt(252)
    sharpe = (cagr - 0.035) / vol if vol > 0 else 0
    dd     = (pf - pf.cummax()) / pf.cummax()
    mdd    = dd.min()

    years = (pf.index[-1] - pf.index[0]).days / 365.25

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
def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.grid(alpha=0.15, color=GRID)


def make_chart(results: dict) -> plt.Figure:
    active = {k: v for k, v in results.items() if v is not None}
    n = len(active)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), facecolor=BG)
    fig.subplots_adjust(hspace=0.4)

    ax1, ax2 = axes

    # 자산가치 vs 투입금
    style_ax(ax1)
    for i, (name, (pf, inv)) in enumerate(active.items()):
        c = PALETTE[i % len(PALETTE)]
        ax1.plot(pf.index, pf / 1e6,  color=c, lw=1.8, label=f"{name} (자산)")
        ax1.plot(inv.index, inv / 1e6, color=c, lw=0.9, ls=":", alpha=0.65,
                 label=f"{name} (투입)")
    ax1.set_title("포트폴리오 자산가치 vs 누적 투입금  (실선=자산, 점선=투입)",
                  fontsize=10, fontweight="bold", color=TEXT)
    ax1.set_ylabel("금액 (백만원)", color=TEXT)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))
    ax1.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=7,
               ncol=min(n * 2, 4), loc="upper left")

    # MDD
    style_ax(ax2)
    for i, (name, (pf, inv)) in enumerate(active.items()):
        c  = PALETTE[i % len(PALETTE)]
        dd = (pf - pf.cummax()) / pf.cummax() * 100
        ax2.fill_between(dd.index, dd, 0, alpha=0.22, color=c)
        ax2.plot(dd.index, dd, color=c, lw=1.1, label=name)
    ax2.set_title("낙폭(Drawdown) 비교", fontsize=10, fontweight="bold", color=TEXT)
    ax2.set_ylabel("낙폭 (%)", color=TEXT)
    ax2.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=7, ncol=n)

    return fig


# ════════════════════════════════════════════════════════════
#  UI 헬퍼: 성과 메트릭 카드
# ════════════════════════════════════════════════════════════
def show_metrics(m: dict, label: str, color: str):
    st.markdown(f"<div style='color:{color};font-weight:bold;font-size:1rem;"
                f"margin-bottom:8px'>▶ {label}</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 투입금",       f"₩{m['total_invested']/1e8:.2f}억")
    c2.metric("최종 자산",       f"₩{m['final_val']/1e8:.2f}억")
    c3.metric("수익금",          f"₩{m['profit']/1e6:+,.0f}만",
              delta=f"{m['total_ret']:+.1f}%")
    c4.metric("CAGR (IRR)",      f"{m['cagr']:+.2f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("연간 변동성",     f"{m['vol']:.2f}%")
    c6.metric("샤프 비율",       f"{m['sharpe']:.2f}")
    c7.metric("최대 낙폭 MDD",   f"{m['mdd']:.2f}%")
    c8.metric("백테스트 기간",   f"{m['years']:.1f}년")
    st.markdown("---")


# ════════════════════════════════════════════════════════════
#  메인 UI
# ════════════════════════════════════════════════════════════
st.title("📊 포트폴리오 백테스터")
st.markdown(
    '<div class="info-box">'
    '달러 자산(IAU·QQQ·UUP) → 일별 KRW=X 환율로 <b>원화 환산</b> ｜ '
    '원화 자산(맥쿼리인프라 088980.KS) → 환산 없음 ｜ '
    '배당 재투자 · 반기 리밸런싱 포함</div>',
    unsafe_allow_html=True
)

# ── 사이드바: 글로벌 설정 ─────────────────────────────────
with st.sidebar:
    st.header("⚙️ 기본 설정")

    col_s, col_e = st.columns(2)
    start_date = col_s.date_input("시작일", value=date(2016, 1, 1),
                                  min_value=date(2010, 1, 1), max_value=date(2024, 1, 1))
    end_date   = col_e.date_input("종료일", value=date(2025, 6, 30),
                                  min_value=date(2011, 1, 1), max_value=date(2025, 12, 31))

    st.markdown("**비중 설정 (%)**")
    w_gold   = st.slider("금현물",    0, 100, 25, 5)
    w_nas    = st.slider("나스닥100", 0, 100, 25, 5)
    w_cash   = st.slider("현금(소파)", 0, 100, 25, 5)
    w_reit   = st.slider("한국리츠",  0, 100, 25, 5)
    total_w  = w_gold + w_nas + w_cash + w_reit
    if total_w != 100:
        st.warning(f"비중 합계: {total_w}% (100%가 아님 → 자동 정규화)")

    weights_raw = {
        "금현물": w_gold, "나스닥100": w_nas,
        "현금(소파)": w_cash, "한국리츠": w_reit,
    }
    weights = {k: v / total_w for k, v in weights_raw.items()}

    st.markdown("**연 배당률 설정 (%)**")
    y_gold  = st.number_input("금현물 배당",    0.0, 20.0, 0.0,  0.1)
    y_nas   = st.number_input("나스닥100 배당", 0.0, 20.0, 3.0,  0.1)
    y_cash  = st.number_input("현금(소파) 배당", 0.0, 20.0, 4.0, 0.1)
    y_reit  = st.number_input("한국리츠 배당",  0.0, 20.0, 6.5,  0.1)
    monthly_yields = {
        "금현물": y_gold/100/12, "나스닥100": y_nas/100/12,
        "현금(소파)": y_cash/100/12, "한국리츠": y_reit/100/12,
    }

    cost = st.slider("리밸런싱 수수료 (%)", 0.0, 2.0, 0.5, 0.05) / 100

    st.markdown("---")
    st.caption("💡 각 시나리오 탭에서 입금 방식을 설정하세요.")

# ── 시나리오 탭 ──────────────────────────────────────────
tab_a, tab_b, tab_c, tab_d = st.tabs([
    "💰 A: 일시납", "📅 B: 월적립", "🔀 C: 혼합", "🛠️ D: 커스텀"
])

scenarios = {}

with tab_a:
    st.subheader("A. 일시납")
    st.markdown("전체 자금을 첫날 한 번에 투입하는 시나리오입니다.")
    en_a  = st.checkbox("이 시나리오 활성화", value=True, key="en_a")
    ini_a = st.number_input("초기 투입금 (원)", 0, 10_000_000_000,
                             100_000_000, 1_000_000, key="ini_a",
                             format="%d")
    st.caption(f"= ₩{ini_a/1e8:.2f}억")
    scenarios["일시납"] = {"enabled": en_a, "initial": ini_a,
                           "dca_amount": 0, "dca_freq": None}

with tab_b:
    st.subheader("B. 월적립 (DCA)")
    st.markdown("첫날부터 매월 일정 금액을 적립하는 시나리오입니다.")
    en_b  = st.checkbox("이 시나리오 활성화", value=True, key="en_b")
    dca_b = st.number_input("월 적립금 (원)", 0, 100_000_000,
                             3_000_000, 500_000, key="dca_b", format="%d")
    st.caption(f"= 월 ₩{dca_b/1e4:.0f}만 → 연 ₩{dca_b*12/1e8:.2f}억")
    scenarios["월적립"] = {"enabled": en_b, "initial": 0,
                           "dca_amount": dca_b, "dca_freq": "MS"}

with tab_c:
    st.subheader("C. 혼합 (목돈 + 월적립)")
    st.markdown("초기 목돈을 투입한 후, 추가로 매월 적립하는 시나리오입니다.")
    en_c  = st.checkbox("이 시나리오 활성화", value=True, key="en_c")
    ini_c = st.number_input("초기 투입금 (원)", 0, 10_000_000_000,
                             50_000_000, 1_000_000, key="ini_c", format="%d")
    dca_c = st.number_input("월 추가 적립금 (원)", 0, 100_000_000,
                             2_000_000, 500_000, key="dca_c", format="%d")
    st.caption(f"초기 ₩{ini_c/1e8:.2f}억 + 월 ₩{dca_c/1e4:.0f}만")
    scenarios["혼합"] = {"enabled": en_c, "initial": ini_c,
                         "dca_amount": dca_c, "dca_freq": "MS"}

with tab_d:
    st.subheader("D. 커스텀 (비정기 입금)")
    st.markdown("초기 투입 외에 특정 날짜에 비정기적으로 입금하는 시나리오입니다.")
    en_d  = st.checkbox("이 시나리오 활성화", value=True, key="en_d")
    ini_d = st.number_input("초기 투입금 (원)", 0, 10_000_000_000,
                             30_000_000, 1_000_000, key="ini_d", format="%d")

    st.markdown("**비정기 입금 일정** (날짜: YYYY-MM-DD, 금액: 원)")
    default_custom = (
        "2022-04-01: 10000000\n"
        "2022-10-01: 15000000\n"
        "2023-03-01: 10000000\n"
        "2023-09-01: 20000000\n"
        "2024-01-01:  5000000"
    )
    custom_text = st.text_area("입금 스케줄 (한 줄에 하나씩)", default_custom,
                               height=150, key="custom_text")

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
    st.caption(f"비정기 입금 합계: ₩{total_custom/1e4:.0f}만 ({len(custom_deposits)}건) "
               f"| 총 투입 예상: ₩{(ini_d + total_custom)/1e8:.2f}억")

    scenarios["커스텀"] = {
        "enabled": en_d,
        "initial": ini_d,
        "dca_amount": 0,
        "dca_freq": None,
        "custom_deposits": custom_deposits,
    }

# ── 실행 버튼 ─────────────────────────────────────────────
st.markdown("---")
run_btn = st.button("🚀 백테스트 실행", use_container_width=True)

if run_btn:
    enabled = {k: v for k, v in scenarios.items() if v.get("enabled", False)}
    if not enabled:
        st.warning("최소 1개 이상의 시나리오를 활성화하세요.")
        st.stop()

    with st.spinner("📡 야후파이낸스 데이터 수신 중..."):
        try:
            prices, fx = fetch_prices(
                DEFAULT_TICKERS, DEFAULT_IS_USD,
                str(start_date), str(end_date)
            )
        except Exception as e:
            st.error(f"데이터 수집 실패: {e}")
            st.stop()

    # 환율 정보
    st.markdown(
        f'<div class="info-box">📈 환율 정보: '
        f'{str(start_date)[:7]} ≈ <b>{fx.iloc[0]:.0f}원</b> → '
        f'{str(end_date)[:7]} ≈ <b>{fx.iloc[-1]:.0f}원</b> '
        f'(변화: {(fx.iloc[-1]/fx.iloc[0]-1)*100:+.1f}%)</div>',
        unsafe_allow_html=True
    )

    results = {}
    with st.spinner("⚙️ 시나리오별 백테스트 계산 중..."):
        for name, cfg in enabled.items():
            pf, inv = run_backtest(prices, weights, monthly_yields, cfg, cost)
            results[name] = (pf, inv)

    # 차트
    st.subheader("📈 시나리오 비교 차트")
    fig = make_chart(results)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # 성과 지표
    st.subheader("📋 시나리오별 성과 요약")
    for i, (name, (pf, inv)) in enumerate(results.items()):
        m = calc_metrics(pf, inv)
        show_metrics(m, name, PALETTE[i % len(PALETTE)])

    # 비교 테이블
    st.subheader("📊 시나리오 한눈 비교")
    rows = []
    for name, (pf, inv) in results.items():
        m = calc_metrics(pf, inv)
        rows.append({
            "시나리오":    name,
            "총 투입금":   f"₩{m['total_invested']/1e8:.2f}억",
            "최종 자산":   f"₩{m['final_val']/1e8:.2f}억",
            "수익금":      f"₩{m['profit']/1e6:+,.0f}만",
            "수익률":      f"{m['total_ret']:+.2f}%",
            "CAGR":        f"{m['cagr']:+.2f}%",
            "변동성":      f"{m['vol']:.2f}%",
            "샤프":        f"{m['sharpe']:.2f}",
            "MDD":         f"{m['mdd']:.2f}%",
        })
    df_cmp = pd.DataFrame(rows).set_index("시나리오")
    st.dataframe(df_cmp, use_container_width=True)

    # JSON 다운로드
    json_out = {}
    for name, (pf, inv) in results.items():
        m    = calc_metrics(pf, inv)
        pf_m = pf.resample("ME").last()
        inv_m = inv.resample("ME").last()
        dd_m  = ((pf - pf.cummax()) / pf.cummax()).resample("ME").last()
        json_out[name] = {
            "metrics": {k: round(v, 2) for k, v in m.items()},
            "dates":     [d.strftime("%Y-%m") for d in pf_m.index],
            "portfolio": [round(v) for v in pf_m.values],
            "invested":  [round(v) for v in inv_m.values],
            "drawdown":  [round(v * 100, 2) for v in dd_m.values],
        }
    json_bytes = json.dumps(json_out, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button(
        label="⬇️ 결과 JSON 다운로드",
        data=json_bytes,
        file_name="backtest_results.json",
        mime="application/json",
        use_container_width=True,
    )

# ── 푸터 ─────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "데이터 출처: Yahoo Finance ｜ 달러 자산(IAU·QQQ·UUP) → KRW=X 환율 원화 환산 ｜ "
    "088980.KS(맥쿼리인프라) → 원화 직접 ｜ 투자 참고용이며 실제 수익을 보장하지 않습니다."
)
