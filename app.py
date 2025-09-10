# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import plotly.graph_objects as go

# =========================
# Helpers (cacheados + formatadores)
# =========================
def fmt_brl(valor):
    """Formata número para padrão monetário brasileiro (R$ 10.000,00)"""
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# =========================
# Config & Style
# =========================
st.set_page_config(page_title="Investboard (AÇÕES & FII's)", layout="wide")
st.markdown("""
<style>
/* leve polimento visual */
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
h1, h2, h3 { margin-top: 0.4rem; }
.metric { text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("📊 InvestBoard — AÇÕES & FII's")

# =========================
# Universo (listas curadas)
# =========================
IBOV = [
    "PETR4","VALE3","ITUB4","BBDC4","BBAS3","WEGE3","PRIO3","ABEV3","B3SA3","SUZB3",
    "ELET3","ELET6","VIVT3","EQTL3","RAIL3","BRAP4","TIMS3","TOTS3","RENT3","KLBN11",
    "TAEE11","ENGI11","UGPA3","MRVE3","LREN3","HAPV3","CSNA3","CMIG4","CPLE6","SLCE3",
    "AZUL4","GOLL4","CYRE3","USIM5","RRRP3","BRFS3","CCRO3","MGLU3","YDUQ3","PCAR3",
    "ARZZ3","CVCB3","SMTO3","NTCO3","PETZ3","JBSS3","MRFG3","MULT3","BRML3","IGTI11"
]
IFIX = [
    "HGLG11","KNRI11","MXRF11","XPML11","XPLG11","VISC11","VGIR11","CPTS11","IRDM11","BTLG11",
    "HGRU11","HGBS11","DEVA11","KNCR11","KNSC11","KNIP11","GGRC11","MCCI11","RBRF11","RBRP11",
    "RECT11","RZTR11","HCTR11","ALZR11","HGRE11","VGHF11","HFOF11","BCFF11","BRCR11","VINO11"
]

UNIVERSE = {"Ações (IBOV)": IBOV, "FIIs (IFIX)": IFIX}

# =========================
# Helpers (cacheados)
# =========================
@st.cache_data(ttl=3600, show_spinner=False)
def normalize_sa(ticker: str) -> str:
    t = ticker.strip().upper()
    return t if t.endswith(".SA") else f"{t}.SA"

@st.cache_data(ttl=3600, show_spinner=False)
def yf_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}

@st.cache_data(ttl=3600, show_spinner=False)
def yf_history(ticker: str, period="1y", interval="1d") -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def yf_dividends(ticker: str) -> pd.Series:
    try:
        s = yf.Ticker(ticker).dividends
        return s if isinstance(s, pd.Series) else pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)

def dy_12m_percent(ticker_sa: str) -> float:
    """DY = soma(dividendos últimos 365 dias) / último preço * 100"""
    divs = yf_dividends(ticker_sa)
    hist = yf_history(ticker_sa, period="1y", interval="1d")
    if divs.empty or hist.empty:
        return 0.0
    cutoff = pd.Timestamp.now(tz=divs.index.tz) - pd.Timedelta(days=365)
    total = float(divs[divs.index >= cutoff].sum())
    price = float(hist["Close"].iloc[-1]) if len(hist) else 0.0
    return round((total / price) * 100, 2) if price > 0 else 0.0


def fii_liquidez_mm(info: dict, last_price: float) -> float:
    """Liquidez diária (R$ milhões) ≈ preço * volume médio (10d) / 1e6"""
    vol = info.get("averageDailyVolume10Day") or info.get("averageDailyVolume3Month") or 0
    liq = (last_price * vol) / 1e6
    return round(float(liq), 2)

def safe_get_pct(info: dict, key: str) -> float:
    """Converte campos do .info que já vêm como fração (ex.: 0.1234) para %"""
    v = info.get(key)
    if v is None: return 0.0
    try:
        return round(float(v) * 100, 2)
    except Exception:
        return 0.0

def safe_float(info: dict, key: str) -> float:
    v = info.get(key)
    try:
        return round(float(v), 4)
    except Exception:
        return 0.0

# =========================
# UI Tabs
# =========================
tab_view, tab_rank, tab_best, tab_buffett = st.tabs([
    "🔎 Visão do ativo",
    "🏆 Ranking (com filtros)",
    "📖 TOP 10 Ativos (ranking fixo)",
    "📘 Método Warren Buffett: A Fórmula do Enriquecimento Real"
])

# -------------------------
# 🔎 VISÃO DO ATIVO
# -------------------------

with tab_view:
    st.warning("""
    ⚠️ **Aviso Importante:**  
    Este dashboard **não é uma recomendação de investimento**.  
    Os cálculos apresentados foram feitos com base em indicadores fundamentalistas
    (P/L, P/VP, ROE, Dividend Yield e Liquidez), apenas para fins **educacionais e de análise**.  

    É fundamental que cada investidor faça suas próprias avaliações e decisões
    antes de investir em qualquer ativo.
    """)
    colA, colB = st.columns([1, 3])
    with colA:
        
        # Agora o universo já vem com ícones
        classe_name = st.selectbox(
            "Classe de Ativo",
            list(UNIVERSE.keys())  # pega direto as chaves com ícone
        )
        tickers = UNIVERSE[classe_name]

        pick = st.selectbox(
            "Ativo (digite para filtrar)",
            options=tickers,
            index=0,
            help="Você pode digitar PETR4, BBAS3, HGLG11…"
        )

    ticker_sa = normalize_sa(pick)
    info = yf_info(ticker_sa)
    hist = yf_history(ticker_sa, period="1y")
    last_price = float(hist["Close"].iloc[-1]) if not hist.empty else float(info.get("currentPrice") or 0)
    dy12 = dy_12m_percent(ticker_sa)
 
    display_ticker = ticker_sa.replace(".SA", "")

    if "FIIs" in classe_name:
        st.subheader(f"🏢 {display_ticker}")
    else:
        st.subheader(f"📈 {display_ticker}")

    if "FIIs" in classe_name:  # métricas para FIIs
        pvp = safe_float(info, "priceToBook")
        if not pvp or pvp == 0.0:
            pvp_display = "Falha no retorno"
        else:
            pvp_display = pvp

        liq = fii_liquidez_mm(info, last_price)
        dy12 = dy_12m_percent(ticker_sa)

        c1, c2, c3 = st.columns(3)
        c1.metric("P/VP", pvp_display)
        c2.metric("Liquidez diária (R$ mi)", liq)
        c3.metric("Dividend Yield (12m)", f"{dy12}%")

    else:  # métricas para Ações
        pl  = safe_float(info, "trailingPE")
        pvp = safe_float(info, "priceToBook")
        roe = safe_get_pct(info, "returnOnEquity")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("P/L", pl)
        c2.metric("P/VP", pvp)
        c3.metric("ROE", f"{roe}%")
        c4.metric("Dividend Yield (12m)", f"{dy12}%")

    # Gráfico de preço
    if not hist.empty:
        st.plotly_chart(
            px.line(hist.reset_index(), x="Date", y="Close", title="📈 Preço — 1 ano"),
            use_container_width=True
        )
    else:
        st.info("Sem histórico recente disponível.")

    # Gráfico de dividendos
    divs = yf_dividends(ticker_sa)
    if not divs.empty:
        df_div = divs.reset_index().rename(columns={"Date": "Data", "Dividends": "Provento"})
        cutoff = pd.Timestamp.now(tz=df_div["Data"].dt.tz) - pd.Timedelta(days=5*365)
        df_div = df_div[df_div["Data"] >= cutoff]
        st.plotly_chart(
            px.bar(df_div, x="Data", y="Provento", title="💰 Dividendos pagos"),
            use_container_width=True
        )
    else:
        st.info("Sem registro de dividendos neste período.")


# -------------------------
# 🏆 RANKING
# -------------------------
with tab_rank:
    classe_name = st.selectbox("Universo para ranking", list(UNIVERSE.keys()), key="rank_uni")
    tickers = UNIVERSE[classe_name]

    rows_all = []  # guarda todos
    rows_pass = [] # guarda só os que passaram

    if "FIIs" in classe_name:
        # Filtros FIIs
        max_pvp = st.slider("P/VP máximo", 0.0, 5.0, 2.0, 0.1)
        min_liq = st.slider("Liquidez diária mínima (R$ mi)", 0.0, 50.0, 2.0, 0.1)
        min_dy  = st.slider("Dividend Yield mínimo (12m, %)", 0.0, 30.0, 8.0, 0.1)

        for t in tickers:
            tsa = normalize_sa(t)
            info = yf_info(tsa)
            hist = yf_history(tsa, period="1y")
            if hist.empty: continue
            last_price = float(hist["Close"].iloc[-1])
            pvp = safe_float(info, "priceToBook")
            liq = fii_liquidez_mm(info, last_price)
            dy  = dy_12m_percent(tsa)

            # score: DY 60% + Liquidez 40%
            score = dy * 0.6 + min(liq, 50) * 0.4
            row = [t, pvp, liq, dy, round(score, 2)]
            rows_all.append(row)

            if (pvp > 0 and pvp <= max_pvp) and (liq >= min_liq) and (dy >= min_dy):
                rows_pass.append(row)

        cols = ["Ticker", "P/VP", "Liq (R$ mi)", "DY (12m, %)", "Score"]

    else:
        # Filtros Ações
        max_pl  = st.slider("P/L máximo", 0.0, 60.0, 25.0, 0.5)
        max_pvp = st.slider("P/VP máximo", 0.0, 8.0, 3.0, 0.1)
        min_roe = st.slider("ROE mínimo (%)", 0.0, 40.0, 8.0, 0.1)
        min_dy  = st.slider("Dividend Yield mínimo (12m, %)", 0.0, 30.0, 3.0, 0.1)

        for t in tickers:
            tsa  = normalize_sa(t)
            info = yf_info(tsa)
            hist = yf_history(tsa, period="1y")
            if hist.empty: continue

            pl   = safe_float(info, "trailingPE")
            pvp  = safe_float(info, "priceToBook")
            roe  = safe_get_pct(info, "returnOnEquity")
            dy   = dy_12m_percent(tsa)

            inv_pl  = min(20.0, 20.0 / pl) if pl > 0 else 0
            inv_pvp = min(10.0, 10.0 / pvp) if pvp > 0 else 0
            score   = dy * 0.4 + roe * 0.3 + inv_pl + inv_pvp
            row = [t, pl, pvp, roe, dy, round(score, 2)]
            rows_all.append(row)

            if (pl > 0 and pl <= max_pl) and (pvp > 0 and pvp <= max_pvp) and (roe >= min_roe) and (dy >= min_dy):
                rows_pass.append(row)

        cols = ["Ticker", "P/L", "P/VP", "ROE (%)", "DY (12m, %)", "Score"]

    # Monta DataFrame final (sempre 10)
    df_pass = pd.DataFrame(rows_pass, columns=cols)
    df_all  = pd.DataFrame(rows_all,  columns=cols)

    if len(df_pass) < 10:
        # completa com melhores do universo
        df = pd.concat([df_pass, df_all]).drop_duplicates(subset=["Ticker"])
        df = df.sort_values("Score", ascending=False).head(10)
    else:
        df = df_pass.sort_values("Score", ascending=False).head(10)

    st.subheader("🥇 Top 10")
    st.dataframe(df, use_container_width=True)

    if not df.empty:
        st.plotly_chart(
            px.bar(df, x="Ticker", y="Score", title="Score dos Top 10"),
            use_container_width=True
        )
    else:
        st.info("Nenhum ativo pôde ser avaliado.")

# -------------------------
# 📑 ESTRATÉGIA & RANKINGS FIXOS
# -------------------------
with tab_best:
    st.subheader("📑 Estratégia usada para ranqueamento")

    st.markdown("""
    O **InvestBoard** classifica os ativos com base em um **Score**,
    que combina múltiplos indicadores para tentar equilibrar **valor** e **qualidade**.
    ...
    """)
    st.markdown("""
    O **InvestBoard** classifica os ativos com base em um **Score**,
    que combina múltiplos indicadores para tentar equilibrar **valor** e **qualidade**.

    ### Para **Ações (IBOV)**
    - **Dividend Yield (DY)** → 40% do peso  
    - **ROE (Retorno sobre Patrimônio)** → 30% do peso  
    - **P/L (Preço/Lucro)** → 20% (quanto menor, melhor, mas com limite)  
    - **P/VP (Preço/Valor Patrimonial)** → 10%  

    ### Para **FIIs (IFIX)**
    - **Dividend Yield (DY)** → 60% do peso  
    - **Liquidez diária** → 40% (limitada em até R$ 50 milhões/dia)  
    - P/VP é usado como filtro adicional, não entra no score.  

    Assim, sempre apresentamos os **Top 10 ativos** em cada categoria.
    """)

    # -------------------
    # Top 10 AÇÕES (fixo)
    # -------------------
    st.subheader("📈 Top 10 Ações (IBOV)")
    rows_a = []
    for t in IBOV:
        tsa  = normalize_sa(t)
        info = yf_info(tsa)
        hist = yf_history(tsa, period="1y")
        if hist.empty: continue

        pl   = safe_float(info, "trailingPE")
        pvp  = safe_float(info, "priceToBook")
        roe  = safe_get_pct(info, "returnOnEquity")
        dy   = dy_12m_percent(tsa)

        inv_pl  = min(20.0, 20.0 / pl) if pl > 0 else 0
        inv_pvp = min(10.0, 10.0 / pvp) if pvp > 0 else 0
        score   = dy * 0.4 + roe * 0.3 + inv_pl + inv_pvp
        rows_a.append([t, pl, pvp, roe, dy, round(score, 2)])

    df_a = pd.DataFrame(rows_a, columns=["Ticker", "P/L", "P/VP", "ROE (%)", "DY (12m, %)", "Score"])
    df_a = df_a.sort_values("Score", ascending=False).head(10)
    st.dataframe(df_a, use_container_width=True)
    st.plotly_chart(px.bar(df_a, x="Ticker", y="Score", title="Top 10 Ações (Score)"), use_container_width=True)

    # -------------------
    # Top 10 FIIs (fixo)
    # -------------------
    st.subheader("🏢 Top 10 FIIs (IFIX)")
    rows_f = []
    for t in IFIX:
        tsa = normalize_sa(t)
        info = yf_info(tsa)
        hist = yf_history(tsa, period="1y")
        if hist.empty: continue
        last_price = float(hist["Close"].iloc[-1])
        pvp = safe_float(info, "priceToBook")
        liq = fii_liquidez_mm(info, last_price)
        dy  = dy_12m_percent(tsa)

        score = dy * 0.6 + min(liq, 50) * 0.4
        rows_f.append([t, pvp if pvp > 0 else "Falha no retorno", liq, dy, round(score, 2)])

    df_f = pd.DataFrame(rows_f, columns=["Ticker", "P/VP", "Liq (R$ mi)", "DY (12m, %)", "Score"])
    df_f = df_f.sort_values("Score", ascending=False).head(10)
    st.dataframe(df_f, use_container_width=True)
    st.plotly_chart(px.bar(df_f, x="Ticker", y="Score", title="Top 10 FIIs (Score)"), use_container_width=True)

    # =========================
# Rodapé
# =========================

# -------------------------
# 📖 Estratégia Buy & Hold
# -------------------------
with tab_buffett:
    st.header("🎶 Compre ao som dos canhões e venda ao som dos violinos")

    # Introdução com frase icônica
    st.markdown("""

    Essa frase, popularizada pelo oráculo de Omaha, sintetiza a ideia de comprar ativos de qualidade em momentos de crise
    e esperar pacientemente o retorno no longo prazo.  

    O **Buy & Hold** aliado à **Análise Fundamentalista** é a estratégia que mais formou
    milionários no mundo moderno, pois se apoia em fundamentos, paciência
    e no poder dos **juros compostos**.
    """)

    # ===================
    # Juros Compostos
    # ===================
    st.subheader("🔑 Juros Compostos (o motor do método)")

    st.latex(r"VF = VP \times (1 + r)^{n}")

    st.markdown("""
    Onde:  
    - **VF** = valor futuro  
    - **VP** = valor inicial  
    - **r** = taxa anual  
    - **n** = anos  

    O reinvestimento contínuo faz com que dividendos e lucros se multipliquem,
    criando crescimento exponencial ao longo do tempo.
    """)

    vp_ex = 10_000
    vf_08 = vp_ex * (1.08**30)
    vf_12 = vp_ex * (1.12**30)
    vf_15 = vp_ex * (1.15**30)

    # Função de formatação no estilo brasileiro
    def fmt_brl(valor):
        return f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    st.markdown(f"""
    **Exemplo (30 anos, investimento inicial de R$ {fmt_brl(vp_ex)}):**  
    - Renda Fixa (8% a.a): ≈ **R$ {fmt_brl(vf_08)}**  
    - Bolsa (12% a.a): ≈ **R$ {fmt_brl(vf_12)}**  
    - Excelentes empresas (15% a.a): ≈ **R$ {fmt_brl(vf_15)}**
    """)


    st.markdown(f"""
    **Exemplo (30 anos, investimento inicial de R$ {vp_ex:,.0f}):**
    - Renda Fixa (8% a.a): ≈ **R$ {vf_08:,.0f}**  
    - Bolsa (12% a.a): ≈ **R$ {vf_12:,.0f}**  
    - Excelentes empresas (15% a.a): ≈ **R$ {vf_15:,.0f}**
    """)

    # ===================
    # Day Trade / Robôs
    # ===================
    st.subheader("📉 Por que Day Trade e Robôs não funcionam no longo prazo")
    st.markdown("""
    - Estudo da **CVM/FGV (2019)** acompanhando milhares de traders mostrou que
      **apenas 0,1%** conseguiu lucro líquido consistente após 2 anos.  
    - Robôs de trade automatizam especulação, mas não eliminam o risco.
    - Estatisticamente, **mais de 95% perdem dinheiro no longo prazo**.
    """)

    st.markdown("""
    <div style="border-left:6px solid #e50914; padding:12px; background:#2a0000; border-radius:6px;">
    <b style="color:#ffcccc;">🚨 Importante:</b> Day Trade é estatisticamente insustentável.
    É um cemitério silencioso: você vê apenas o iceberg dos poucos que ganham,
    mas a base enorme de quem perde fica oculta.
    </div>
    """, unsafe_allow_html=True)

    # ===================
    # Renda Variável vs Renda Fixa
    # ===================
    st.subheader("📊 Por que Renda Variável supera Renda Fixa no longo prazo")
    st.markdown("""
    - O **UBS Global Investment Returns Yearbook (2024)** mostra que,
      em 123 anos de dados globais, as ações renderam em média **5,3% acima da inflação**,
      contra **0,8% de títulos públicos**.  
    - No Brasil, de 2003 a 2023, o **IBOV acumulou +900%**, enquanto o CDI ficou em ~+300%.  
    - A renda fixa preserva o capital, mas **não gera enriquecimento**.
    """)

    # ===================
    # Imóveis vs Bolsa
    # ===================
    
    st.subheader("🏠 Por que imóveis perderam espaço frente à Bolsa")
    
    st.markdown("""
    - Décadas atrás, terrenos e imóveis multiplicavam valor com urbanização acelerada.  
    - Hoje, valorizam-se em linha com a inflação (~6% a.a), e ainda têm custos altos
      (manutenção, IPTU, corretagem, vacância).  
    - A Bolsa (ações + FIIs) rende em média **12% a.a**, com liquidez e dividendos.  
    - Exemplo: um imóvel de 200 mil reais tende a valer 1,15 milhão reais em 30 anos (~6% a.a).  
      O mesmo valor na Bolsa a 12% a.a pode chegar a R$ 6 milhões.  
    """)

    # ===================
    # Cases de Sucesso
    # ===================
    st.subheader("🌍 Casos de sucesso com Buy & Hold")

    st.markdown("""
    - **Warren Buffett (EUA):** começou comprando ações aos 11 anos, transformou a
      Berkshire Hathaway em um império de **US$ 700 bilhões**.  
    - **Peter Lynch (EUA):** gestor do fundo Magellan (Fidelity) entre 1977 e 1990,
      entregou **29% a.a** durante 13 anos, multiplicando capital de investidores.  
    - **Luiz Barsi (Brasil):** maior investidor pessoa física da B3.  
      Filho de imigrantes pobres, começou comprando ações nos anos 60 e
      construiu fortuna bilionária apenas com **dividendos reinvestidos**.  
    - **Estudos acadêmicos (Fama & French, 1993):** confirmam que fatores fundamentais
      (lucro, valor contábil, dividendos) explicam retornos de longo prazo muito mais
      do que o acaso ou análises gráficas.
    """)

    # ===================
    # Gráficos comparativos
    # ===================
    anos = list(range(0, 31))
    def curva(vp, r): return [vp * ((1 + r) ** n) for n in anos]

    buy_hold_15 = curva(10_000, 0.15)
    day_trade_07 = curva(10_000, 0.07)
    bolsa_12 = curva(10_000, 0.12)
    rf_08 = curva(10_000, 0.08)
    bolsa_12_im = curva(200_000, 0.12)
    imovel_06 = curva(200_000, 0.06)

    # Gráfico 1: Buy & Hold vs Day Trade
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=anos, y=buy_hold_15, name='Buy & Hold (15% a.a)', line=dict(color='blue', width=3)))
    fig1.add_trace(go.Scatter(x=anos, y=day_trade_07, name='Day Trade / Robôs (~7% a.a)', line=dict(color='gold', width=3)))
    fig1.update_layout(title="📊 Buy & Hold vs Day Trade (R$ 10.000 por 30 anos)", template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)

    # Gráfico 2: Renda Fixa vs Bolsa
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=anos, y=bolsa_12, name='Bolsa (~12% a.a)', line=dict(color='blue', width=3)))
    fig2.add_trace(go.Scatter(x=anos, y=rf_08, name='Renda Fixa (~8% a.a)', line=dict(color='gold', width=3)))
    fig2.update_layout(title="💵 Renda Fixa vs Bolsa (R$ 10.000 por 30 anos)", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

    # Gráfico 3: Imóveis vs Bolsa
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=anos, y=bolsa_12_im, name='Bolsa (Ações/FIIs ~12% a.a)', line=dict(color='blue', width=3)))
    fig3.add_trace(go.Scatter(x=anos, y=imovel_06, name='Imóveis (~6% a.a)', line=dict(color='gold', width=3)))
    fig3.update_layout(title="🏠 Imóveis vs Bolsa (R$ 200.000 por 30 anos)", template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

    # ===================
    # Conclusão
    # ===================
    st.success("""
    ✅ Conclusão:  
    - **Day Trade/robôs:** estatísticas provam que mais de 95% perdem no longo prazo.  
    - **Renda Fixa:** protege, mas não enriquece.  
    - **Imóveis:** bons para diversificação, mas pouco rentáveis e ilíquidos.  
    - **Buy & Hold:** comprovadamente o caminho mais sólido para multiplicar patrimônio,
      responsável por criar **Buffett, Lynch e Barsi** — e milhares de milionários comuns.  
    """)