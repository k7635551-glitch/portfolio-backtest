"""
포트폴리오 백테스트 웹앱 — 금융권 UI 버전
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
from matplotlib import font_manager as fm
import warnings
import json
import requests
from datetime import date

warnings.filterwarnings("ignore")

# ── 한글 폰트: 파일 직접 로드 ─────────────────────────────
_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_KR_FONT   = fm.FontProperties(fname=_FONT_PATH, size=9)
_KR_FONT_S = fm.FontProperties(fname=_FONT_PATH, size=8)
_KR_FONT_T = fm.FontProperties(fname=_FONT_PATH, size=10)
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
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, p, span, div {
    font-family: 'Noto Sans KR', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif !important;
}

/* 배경 */
.stApp { background-color: #f5f6f8; }
section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e3ea; }

/* 메트릭 카드 */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e0e3ea;
    border-radius: 6px;
    padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
[data-testid="metric-container"] label {
    font-size: 0.75rem !important;
    color: #6b7280 !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: #111827 !important;
}

/* 버튼 */
.stButton > button {
    background: #003087;
    color: #ffffff;
    font-weight: 600;
    border: none;
    border-radius: 4px;
    padding: 0.55rem 1.2rem;
    font-size: 0.9rem;
    letter-spacing: 0.02em;
    transition: background 0.15s;
    width: 100%;
}
.stButton > button:hover { background: #004db3; }

/* 탭 */
.stTabs [data-baseweb="tab-list"] { border-bottom: 2px solid #e0e3ea; gap: 0; }
.stTabs [data-baseweb="tab"] {
    font-size: 0.85rem;
    font-weight: 500;
    color: #6b7280;
    padding: 8px 18px;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
}
.stTabs [aria-selected="true"] {
    color: #003087 !important;
    border-bottom: 2px solid #003087 !important;
    font-weight: 700 !important;
}

/* 구분선 */
hr { border: none; border-top: 1px solid #e0e3ea; margin: 20px 0; }

/* 결과 행 */
.result-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 9px 0;
    border-bottom: 1px solid #f0f1f4;
    font-size: 0.9rem;
}
.result-row:last-child { border-bottom: none; }
.result-label { color: #6b7280; font-weight: 400; }
.result-value { color: #111827; font-weight: 600; }
.result-value.positive { color: #0066cc; }
.result-value.negative { color: #cc0000; }

/* 시나리오 헤더 */
.scenario-header {
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #003087;
    padding: 4px 0 8px 0;
    border-bottom: 2px solid #003087;
    margin-bottom: 12px;
}

/* 카드 */
.card {
    background: #ffffff;
    border: 1px solid #e0e3ea;
    border-radius: 6px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* 리포트 */
.report-box {
    background: #ffffff;
    border: 1px solid #e0e3ea;
    border-left: 3px solid #003087;
    border-radius: 6px;
    padding: 24px 28px;
    margin-top: 20px;
    line-height: 1.8;
    font-size: 0.92rem;
    color: #374151;
}

/* 정보 토글 */
.stExpander { border: 1px solid #e0e3ea !important; border-radius: 6px !important; background: #fff; }

/* 사이드바 섹션 */
.sidebar-section {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #9ca3af;
    margin: 18px 0 8px 0;
}

/* 테이블 */
.stDataFrame { border: 1px solid #e0e3ea; border-radius: 6px; overflow: hidden; }

/* 성공/경고 */
.stSuccess, .stWarning { border-radius: 4px; font-size: 0.85rem; }

@media (max-width: 768px) {
    h1 { font-size: 1.3rem !important; }
    .card { padding: 14px 16px; }
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  유틸: 금액 표기
# ════════════════════════════════════════════════════════════
def fmt_krw(v: float) -> str:
    """₩1억 2,345만 형식"""
    sign  = "-" if v < 0 else ""
    v     = abs(v)
    uk    = int(v // 1e8)
    man   = int((v % 1e8) // 1e4)
    if uk > 0 and man > 0:
        return f"{sign}₩{uk:,}억 {man:,}만"
    elif uk > 0:
        return f"{sign}₩{uk:,}억"
    else:
        return f"{sign}₩{man:,}만"

def fmt_krw_signed(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return sign + fmt_krw(v)


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

PALETTE  = ["#003087", "#0066cc", "#5b9bd5", "#003366"]
BG       = "#f5f6f8"
PANEL    = "#ffffff"
TEXT_COL = "#111827"
GRID_COL = "#e8eaee"
MUTED    = "#6b7280"


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

    fx_raw = yf.download("KRW=X", start=start, end=end,
                         auto_adjust=False, progress=False)["Close"]
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
        fd = trade_index[0]
        deposits[fd] = deposits.get(fd, 0) + cfg["initial"]
    if cfg.get("dca_amount", 0) > 0 and cfg.get("dca_freq"):
        for d in pd.date_range(trade_index[0], trade_index[-1], freq=cfg["dca_freq"]):
            fut = trade_index[trade_index >= d]
            if len(fut):
                deposits[fut[0]] = deposits.get(fut[0], 0) + cfg["dca_amount"]
    for ds, amt in cfg.get("custom_deposits", {}).items():
        fut = trade_index[trade_index >= pd.Timestamp(ds)]
        if len(fut):
            deposits[fut[0]] = deposits.get(fut[0], 0) + amt
    return pd.Series(deposits).sort_index() if deposits else pd.Series(dtype=float)


# ════════════════════════════════════════════════════════════
#  백테스트 엔진
# ════════════════════════════════════════════════════════════
def run_backtest(prices, weights, monthly_yields, cfg, cost):
    names  = list(weights.keys())
    w      = np.array([weights[n] for n in names])
    my_arr = np.array([monthly_yields[n] for n in names])
    dep    = build_deposit_schedule(cfg, prices.index)

    rebal = set()
    for yr in prices.index.year.unique():
        for mo in [1, 7]:
            mask = (prices.index.year == yr) & (prices.index.month == mo)
            if mask.any():
                rebal.add(prices.index[mask][0])

    shares, cash, invested = np.zeros(len(names)), 0.0, 0.0
    pv, iv, prev_mo = [], [], None

    for dt, row in prices.iterrows():
        pa = row[names].values.astype(float)
        sp = np.where(pa > 0, pa, 1.0)

        if dt.month != prev_mo:
            shares += (shares * pa) * my_arr / sp
            prev_mo = dt.month

        if len(dep) > 0 and dt in dep.index:
            cash    += dep[dt]
            invested += dep[dt]
            shares  += (cash * w) / sp
            cash     = 0.0

        if dt in rebal and shares.sum() > 0:
            cv  = (shares * pa).sum()
            tc  = np.abs(cv * w - shares * pa).sum() * cost
            shares = ((cv - tc) * w) / sp

        pv.append((shares * pa).sum() + cash)
        iv.append(invested)

    return pd.Series(pv, index=prices.index), pd.Series(iv, index=prices.index)


# ════════════════════════════════════════════════════════════
#  성과 지표
# ════════════════════════════════════════════════════════════
def calc_metrics(pf, inv):
    ti  = inv.iloc[-1]
    fv  = pf.iloc[-1]
    ret = (fv - ti) / ti if ti > 0 else 0

    diff = inv.diff().fillna(0)
    diff.iloc[0] = inv.iloc[0]
    cf = (diff[diff > 0] * -1.0).copy()
    cf[pf.index[-1]] = cf.get(pf.index[-1], 0) + fv
    full = pd.Series(0.0, index=pf.index)
    for d, v in cf.sort_index().items():
        full[d] += v

    irr  = npf.irr(full.values)
    cagr = (1 + irr) ** 252 - 1 if not np.isnan(irr) else 0

    dep_days = set(diff[diff > 0].index)
    twr = pd.Series(0.0, index=pf.index)
    prev = pf.shift(1).bfill()
    for d in pf.index:
        if d not in dep_days and prev[d] > 0:
            twr[d] = pf.diff().fillna(0)[d] / prev[d]

    vol    = twr.std() * np.sqrt(252)
    sharpe = (cagr - 0.035) / vol if vol > 0 else 0
    mdd    = ((pf - pf.cummax()) / pf.cummax()).min()
    years  = (pf.index[-1] - pf.index[0]).days / 365.25

    return dict(ti=ti, fv=fv, profit=fv-ti, ret=ret*100,
                cagr=cagr*100, vol=vol*100, sharpe=sharpe,
                mdd=mdd*100, years=years)


# ════════════════════════════════════════════════════════════
#  차트 (fontproperties 직접 지정)
# ════════════════════════════════════════════════════════════
def make_chart(results):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7),
                                    facecolor=PANEL,
                                    gridspec_kw={"hspace": 0.5})

    for ax in (ax1, ax2):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_COL)
        ax.grid(alpha=0.4, color=GRID_COL, linewidth=0.6)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontproperties(_KR_FONT_S)

    for i, (name, (pf, inv)) in enumerate(results.items()):
        c = PALETTE[i % len(PALETTE)]
        ax1.plot(pf.index,  pf  / 1e8, color=c, lw=1.8, label=name)
        ax1.plot(inv.index, inv / 1e8, color=c, lw=0.8, ls="--", alpha=0.55)

    ax1.set_title("자산가치 추이  (실선: 자산,  점선: 투입금)",
                  fontproperties=_KR_FONT_T, color=TEXT_COL, pad=10)
    ax1.set_ylabel("금액 (억원)", fontproperties=_KR_FONT, color=MUTED)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}억"))
    for lbl in ax1.yaxis.get_majorticklabels():
        lbl.set_fontproperties(_KR_FONT_S)
    leg1 = ax1.legend(facecolor=PANEL, edgecolor=GRID_COL, fontsize=8, ncol=len(results))
    for t in leg1.get_texts():
        t.set_fontproperties(_KR_FONT_S)

    for i, (name, (pf, inv)) in enumerate(results.items()):
        c  = PALETTE[i % len(PALETTE)]
        dd = (pf - pf.cummax()) / pf.cummax() * 100
        ax2.fill_between(dd.index, dd, 0, alpha=0.15, color=c)
        ax2.plot(dd.index, dd, color=c, lw=1.2, label=name)

    ax2.set_title("낙폭(Drawdown) 비교",
                  fontproperties=_KR_FONT_T, color=TEXT_COL, pad=10)
    ax2.set_ylabel("낙폭 (%)", fontproperties=_KR_FONT, color=MUTED)
    for lbl in ax2.yaxis.get_majorticklabels():
        lbl.set_fontproperties(_KR_FONT_S)
    leg2 = ax2.legend(facecolor=PANEL, edgecolor=GRID_COL, fontsize=8, ncol=len(results))
    for t in leg2.get_texts():
        t.set_fontproperties(_KR_FONT_S)

    fig.patch.set_facecolor(PANEL)
    return fig


# ════════════════════════════════════════════════════════════
#  결과 텍스트 블록
# ════════════════════════════════════════════════════════════
def result_row_html(label, value, cls=""):
    return (f'<div class="result-row">'
            f'<span class="result-label">{label}</span>'
            f'<span class="result-value {cls}">{value}</span>'
            f'</div>')

def show_scenario_results(results):
    cols = st.columns(len(results))
    for i, (name, (pf, inv)) in enumerate(results.items()):
        m = calc_metrics(pf, inv)
        with cols[i]:
            profit_cls = "positive" if m["profit"] >= 0 else "negative"
            cagr_cls   = "positive" if m["cagr"]   >= 0 else "negative"
            html = f'<div class="card">'
            html += f'<div class="scenario-header">{name}</div>'
            html += result_row_html("총 투입금",    fmt_krw(m["ti"]))
            html += result_row_html("최종 자산",    fmt_krw(m["fv"]))
            html += result_row_html("수익금",       fmt_krw_signed(m["profit"]), profit_cls)
            html += result_row_html("누적 수익률",  f'{m["ret"]:+.2f}%',        profit_cls)
            html += result_row_html("CAGR (IRR)",   f'{m["cagr"]:+.2f}%',       cagr_cls)
            html += result_row_html("연간 변동성",  f'{m["vol"]:.2f}%')
            html += result_row_html("샤프 비율",    f'{m["sharpe"]:.2f}')
            html += result_row_html("최대 낙폭",    f'{m["mdd"]:.2f}%',          "negative")
            html += result_row_html("백테스트 기간", f'{m["years"]:.1f}년')
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  비교 테이블
# ════════════════════════════════════════════════════════════
def show_comparison_table(results):
    rows = []
    for name, (pf, inv) in results.items():
        m = calc_metrics(pf, inv)
        rows.append({
            "시나리오":   name,
            "총 투입금":  fmt_krw(m["ti"]),
            "최종 자산":  fmt_krw(m["fv"]),
            "수익금":     fmt_krw_signed(m["profit"]),
            "수익률":     f'{m["ret"]:+.2f}%',
            "CAGR":       f'{m["cagr"]:+.2f}%',
            "변동성":     f'{m["vol"]:.2f}%',
            "샤프":       f'{m["sharpe"]:.2f}',
            "MDD":        f'{m["mdd"]:.2f}%',
        })
    st.dataframe(
        pd.DataFrame(rows).set_index("시나리오"),
        use_container_width=True
    )


# ════════════════════════════════════════════════════════════
#  AI 리포트
# ════════════════════════════════════════════════════════════
def generate_report(results, weights, monthly_yields, cost, start_date, end_date):
    lines = []
    for name, (pf, inv) in results.items():
        m = calc_metrics(pf, inv)
        lines.append(
            f"[{name}] 투입:{fmt_krw(m['ti'])} / 최종:{fmt_krw(m['fv'])} / "
            f"수익률:{m['ret']:+.1f}% / CAGR:{m['cagr']:+.2f}% / "
            f"변동성:{m['vol']:.1f}% / 샤프:{m['sharpe']:.2f} / MDD:{m['mdd']:.1f}%"
        )

    weight_str = " / ".join(f"{k} {v*100:.0f}%" for k, v in weights.items())
    yield_str  = " / ".join(f"{k} 연{v*12*100:.1f}%" for k, v in monthly_yields.items())

    prompt = f"""당신은 자산운용사 수석 리서치 애널리스트입니다.
아래 포트폴리오 백테스트 결과를 바탕으로 기관 투자자 수준의 분석 리포트를 한국어로 작성해 주세요.

[백테스트 조건]
- 기간: {start_date} ~ {end_date}
- 자산 구성: {weight_str}
- 종목: 금현물(IAU), 나스닥100(QQQ), 달러소파(UUP), 한국리츠-맥쿼리인프라(088980.KS)
- 달러 자산: 일별 원달러 환율 반영 (원화 기준 수익률)
- 배당 재투자: {yield_str}
- 반기 리밸런싱 (1월·7월 첫 거래일), 거래 수수료 {cost*100:.2f}%

[시나리오별 성과]
{chr(10).join(lines)}

아래 구조로 리포트를 작성해 주세요. 각 섹션은 마크다운 헤더(##)를 사용하고, 구체적인 수치를 인용하면서 전문적으로 서술해 주세요.

## 1. 핵심 요약
## 2. 시나리오별 분석
## 3. 위험-수익 특성 분석 (변동성·샤프·MDD 중심)
## 4. 환율 및 자산 배분 효과
## 5. 투자자 유형별 전략 제언
## 6. 한계 및 유의사항"""

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"Content-Type": "application/json"},
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    return resp.json()["content"][0]["text"]


# ════════════════════════════════════════════════════════════
#  메인 UI
# ════════════════════════════════════════════════════════════
st.markdown("## 포트폴리오 백테스터")

with st.expander("이 앱에 대하여"):
    st.markdown("""
**포트폴리오 구성** — 금현물(IAU) · 나스닥100(QQQ) · 달러소파(UUP) · 한국리츠 맥쿼리인프라(088980.KS)의 4자산 분산 포트폴리오 과거 성과를 시뮬레이션합니다.

**환율 반영** — 달러 자산(IAU·QQQ·UUP)은 Yahoo Finance KRW=X 일별 환율로 원화 환산하여 실제 한국 투자자 관점의 수익률을 계산합니다.

**백테스트 가정** — 매월 배당 재투자, 1월·7월 반기 리밸런싱, 거래 수수료 적용이 포함됩니다.

**사용 방법** — ① 사이드바에서 기간·비중·배당률 설정 → ② 아래 탭에서 시나리오 설정 → ③ 실행 버튼 클릭
""")

# ── 사이드바 ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("**포트폴리오 설정**")

    st.markdown('<div class="sidebar-section">백테스트 기간</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    start_date = c1.date_input("시작", value=date(2016, 1, 1),
                                min_value=date(2010, 1, 1), max_value=date(2024, 1, 1),
                                label_visibility="collapsed")
    end_date   = c2.date_input("종료", value=date(2025, 6, 30),
                                min_value=date(2011, 1, 1), max_value=date(2025, 12, 31),
                                label_visibility="collapsed")
    st.caption(f"{start_date} ~ {end_date}")

    st.markdown('<div class="sidebar-section">자산 비중 (%)</div>', unsafe_allow_html=True)
    w_gold = st.slider("금현물",     0, 100, 25, 5, key="wg")
    w_nas  = st.slider("나스닥100",  0, 100, 25, 5, key="wn")
    w_cash = st.slider("현금(소파)", 0, 100, 25, 5, key="wc")
    w_reit = st.slider("한국리츠",   0, 100, 25, 5, key="wr")
    total_w = w_gold + w_nas + w_cash + w_reit
    if total_w == 0:
        st.error("비중 합계가 0입니다.")
        st.stop()
    elif total_w != 100:
        st.warning(f"합계 {total_w}% → 자동 정규화")
    weights = {k: v/total_w for k, v in
               {"금현물": w_gold, "나스닥100": w_nas,
                "현금(소파)": w_cash, "한국리츠": w_reit}.items()}

    st.markdown('<div class="sidebar-section">연간 배당/분배금률 (%)</div>', unsafe_allow_html=True)
    y_g = st.number_input("금현물",     0.0, 20.0, 0.0, 0.1, key="yg")
    y_n = st.number_input("나스닥100",  0.0, 20.0, 3.0, 0.1, key="yn")
    y_c = st.number_input("현금(소파)", 0.0, 20.0, 4.0, 0.1, key="yc")
    y_r = st.number_input("한국리츠",   0.0, 20.0, 6.5, 0.1, key="yr")
    monthly_yields = {
        "금현물": y_g/100/12, "나스닥100": y_n/100/12,
        "현금(소파)": y_c/100/12, "한국리츠": y_r/100/12,
    }

    st.markdown('<div class="sidebar-section">리밸런싱 수수료</div>', unsafe_allow_html=True)
    cost = st.slider("수수료 (%)", 0.0, 2.0, 0.5, 0.05, key="cost") / 100

# ── 시나리오 탭 ───────────────────────────────────────────
st.markdown("#### 입금 시나리오")
tab_a, tab_b, tab_c, tab_d = st.tabs(["A. 일시납", "B. 월적립", "C. 혼합", "D. 커스텀"])
scenarios = {}

with tab_a:
    st.caption("전체 자금을 첫날 한 번에 투입합니다. 투자 시점의 영향을 가장 직접적으로 확인할 수 있습니다.")
    en_a  = st.checkbox("활성화", value=True, key="en_a")
    ini_a = st.number_input("초기 투입금 (원)", 0, 10_000_000_000,
                             100_000_000, 1_000_000, key="ini_a", format="%d")
    st.caption(f"= {fmt_krw(ini_a)}")
    scenarios["일시납"] = {"enabled": en_a, "initial": ini_a, "dca_amount": 0, "dca_freq": None}

with tab_b:
    st.caption("매월 일정 금액을 적립합니다. 매입 단가 평균화(DCA) 효과를 확인할 수 있습니다.")
    en_b  = st.checkbox("활성화", value=True, key="en_b")
    dca_b = st.number_input("월 적립금 (원)", 0, 100_000_000,
                             3_000_000, 500_000, key="dca_b", format="%d")
    st.caption(f"월 {fmt_krw(dca_b)} × 12개월 = 연 {fmt_krw(dca_b*12)}")
    scenarios["월적립"] = {"enabled": en_b, "initial": 0, "dca_amount": dca_b, "dca_freq": "MS"}

with tab_c:
    st.caption("초기 목돈 투입 후 매월 추가 적립합니다. 두 전략의 시너지를 확인할 수 있습니다.")
    en_c  = st.checkbox("활성화", value=True, key="en_c")
    ini_c = st.number_input("초기 투입금 (원)", 0, 10_000_000_000,
                             50_000_000, 1_000_000, key="ini_c", format="%d")
    dca_c = st.number_input("월 적립금 (원)", 0, 100_000_000,
                             2_000_000, 500_000, key="dca_c", format="%d")
    st.caption(f"초기 {fmt_krw(ini_c)} + 월 {fmt_krw(dca_c)}")
    scenarios["혼합"] = {"enabled": en_c, "initial": ini_c, "dca_amount": dca_c, "dca_freq": "MS"}

with tab_d:
    st.caption("비정기 입금 패턴을 직접 설정합니다. 보너스·퇴직금 등 실제 현금흐름을 반영할 수 있습니다.")
    en_d  = st.checkbox("활성화", value=True, key="en_d")
    ini_d = st.number_input("초기 투입금 (원)", 0, 10_000_000_000,
                             30_000_000, 1_000_000, key="ini_d", format="%d")
    st.caption("날짜: 금액 형식으로 한 줄에 하나씩 입력하세요. (예: 2023-03-01: 10000000)")
    default_custom = (
        "2022-04-01: 10000000\n2022-10-01: 15000000\n"
        "2023-03-01: 10000000\n2023-09-01: 20000000\n2024-01-01: 5000000"
    )
    custom_text = st.text_area("입금 일정", default_custom, height=140, key="ct")
    custom_deposits = {}
    for line in custom_text.strip().split("\n"):
        try:
            p = line.replace(",", "").split(":")
            custom_deposits[p[0].strip()] = float(p[1].strip())
        except Exception:
            pass
    total_cd = sum(custom_deposits.values())
    st.caption(f"비정기 입금 합계: {fmt_krw(total_cd)} ({len(custom_deposits)}건) | "
               f"총 예상 투입: {fmt_krw(ini_d + total_cd)}")
    scenarios["커스텀"] = {
        "enabled": en_d, "initial": ini_d, "dca_amount": 0,
        "dca_freq": None, "custom_deposits": custom_deposits,
    }

# ── 실행 ─────────────────────────────────────────────────
st.markdown("---")
run_btn = st.button("백테스트 실행")

if run_btn:
    enabled = {k: v for k, v in scenarios.items() if v.get("enabled")}
    if not enabled:
        st.warning("하나 이상의 시나리오를 활성화하세요.")
        st.stop()

    with st.spinner("시장 데이터를 수신하고 있습니다..."):
        try:
            prices, fx = fetch_prices(
                DEFAULT_TICKERS, DEFAULT_IS_USD,
                str(start_date), str(end_date)
            )
        except Exception as e:
            st.error(f"데이터 수집 실패: {e}")
            st.stop()

    fx_chg = (fx.iloc[-1] / fx.iloc[0] - 1) * 100
    st.info(
        f"원달러 환율: {str(start_date)[:7]} {fx.iloc[0]:.0f}원 → "
        f"{str(end_date)[:7]} {fx.iloc[-1]:.0f}원  ({fx_chg:+.1f}%)  "
        f"— 달러 자산 수익률에 환율 변동이 반영되어 있습니다."
    )

    with st.spinner("백테스트를 계산하고 있습니다..."):
        results = {}
        for name, cfg in enabled.items():
            pf, inv = run_backtest(prices, weights, monthly_yields, cfg, cost)
            results[name] = (pf, inv)

    # 1. 차트
    st.markdown("#### 성과 차트")
    try:
        fig = make_chart(results)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as e:
        st.warning(f"차트 렌더링 실패: {e}")

    # 2. 시나리오별 성과 카드
    st.markdown("#### 시나리오별 성과")
    show_scenario_results(results)

    # 3. 비교 테이블
    st.markdown("#### 시나리오 한눈 비교")
    show_comparison_table(results)

    # 4. AI 해석 리포트 (자동 생성)
    st.markdown("#### AI 해석 리포트")
    with st.spinner("분석 리포트를 생성하고 있습니다..."):
        try:
            report = generate_report(results, weights, monthly_yields,
                                     cost, start_date, end_date)
            st.markdown(
                f'<div class="report-box">{report.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True
            )
        except Exception as e:
            st.warning(f"리포트 생성 실패: {e}")

    # 5. JSON 다운로드
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
        label="결과 JSON 다운로드",
        data=json.dumps(json_out, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="backtest_results.json",
        mime="application/json",
    )

# ── 푸터 ─────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "데이터 출처: Yahoo Finance  |  달러 자산(IAU·QQQ·UUP): KRW=X 환율 원화 환산  |  "
    "088980.KS(맥쿼리인프라): 원화 직접  |  본 서비스는 투자 참고 목적이며 수익을 보장하지 않습니다."
)
