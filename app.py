import io
import math
import textwrap
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

st.set_page_config(page_title="Auto Stats App", layout="wide")

# --- Sidebar e "Sobre" ---
st.sidebar.title("Menu")
if st.sidebar.button("Sobre"):
    st.sidebar.info("Desenvolvido por Dr Fernando Freua - fernando.freua@hc.fm.usp.br - A distribuição é gratuita")


st.title("📊 Auto Stats: Análises Estatísticas Automáticas")
st.caption("Faça upload de um arquivo CSV/Excel com dados tabulados. O app detecta tipos de variáveis, sugere análises e gera gráficos e um PDF.")

uploaded = st.file_uploader("Envie seu arquivo (.csv, .xlsx, .xls)", type=["csv", "xlsx", "xls"])

@st.cache_data(show_spinner=False)
def load_data(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

def detect_types(df: pd.DataFrame):
    """Classifica colunas em numericas, categóricas e datetimes"""
    types = {}
    for col in df.columns:
        s = df[col]
        # Tenta datetime
        if pd.api.types.is_datetime64_any_dtype(s):
            types[col] = "datetime"
            continue
        # Tenta strings com formato de data
        if s.dtype == object:
            try:
                pd.to_datetime(s, errors="raise")
                types[col] = "datetime"
                continue
            except Exception:
                pass
        # Heuristica numerica Vs Categórica
        if pd.api.types.is_numeric_dtype(s):
            # many unique => numeric; few unique => maybe categorical numeric labels
            nunique = s.nunique(dropna=True)
            if nunique <= max(10, int(len(s) * 0.05)):
                types[col] = "categorical"
            else:
                types[col] = "numeric"
        else:
            # text/object -> categorical if limited unique, else text
            nunique = s.nunique(dropna=True)
            types[col] = "categorical" if nunique <= max(30, int(len(s) * 0.1)) else "text"
    return types

def summarize_descriptive(df: pd.DataFrame, types: dict):
    numeric_cols = [c for c,t in types.items() if t=="numeric"]
    cat_cols = [c for c,t in types.items() if t=="categororical"]
    dt_cols = [c for c,t in types.items() if t=="datetime"]

    summaries = {}

    if numeric_cols:
        desc = df[numeric_cols].describe().T
        desc["missing"] = df[numeric_cols].isna().sum()
        summaries["numeric"] = desc

    if cat_cols:
        freq = {}
        for c in cat_cols:
            vc = df[c].value_counts(dropna=False).head(20)
            freq[c] = vc
        summaries["categorical"] = freq

    if dt_cols:
        dt_sum = {}
        for c in dt_cols:
            s = pd.to_datetime(df[c], errors="coerce")
            dt_sum[c] = {
                "min": s.min(),
                "max": s.max(),
                "missing": s.isna().sum()
            }
        summaries["datetime"] = pd.DataFrame(dt_sum).T

    return summaries

def suggest_inferential_tests(df: pd.DataFrame, types: dict):
    """
    Heurística ("IA local") para sugerir testes:
    - num ~ cat(2 níveis) -> t-test
    - num ~ cat(>2 níveis) -> ANOVA
    - cat ~ cat -> Qui-quadrado
    - num ~ num -> Correlação (Pearson + Spearman)
    """
    suggestions = []
    numeric_cols = [c for c,t in types.items() if t=="numeric"]
    cat_cols = [c for c,t in types.items() if t=="categorical"]

    # num ~ cat
    for y in numeric_cols:
        for g in cat_cols:
            levels = df[g].dropna().unique()
            k = len(levels)
            if k == 2:
                suggestions.append(("t-test", y, g))
            elif k > 2 and k <= 10:
                suggestions.append(("anova", y, g))

    # cat ~ cat
    for i, a in enumerate(cat_cols):
        for b in cat_cols[i+1:]:
            suggestions.append(("chi2", a, b))

    # num ~ num
    for i, a in enumerate(numeric_cols):
        for b in numeric_cols[i+1:]:
            suggestions.append(("correlation", a, b))

    # deduplicate
    seen = set()
    uniq = []
    for s in suggestions:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq[:25]  # limit

def run_test(df, kind, a, b=None):
    res = {"test": kind, "a": a, "b": b}
    try:
        if kind == "t-test":
            groups = [g.dropna().values for _, g in df.groupby(b)[a]]
            if len(groups) == 2:
                stat, p = stats.ttest_ind(groups[0], groups[1], equal_var=False, nan_policy="omit")
                res.update({"stat": stat, "p_value": p})
        elif kind == "anova":
            groups = [g.dropna().values for _, g in df.groupby(b)[a]]
            if len(groups) >= 2:
                stat, p = stats.f_oneway(*groups)
                res.update({"stat": stat, "p_value": p, "k_groups": len(groups)})
        elif kind == "chi2":
            ct = pd.crosstab(df[a], df[b])
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            res.update({"chi2": chi2, "p_value": p, "dof": dof})
        elif kind == "correlation":
            s1 = pd.to_numeric(df[a], errors="coerce")
            s2 = pd.to_numeric(df[b], errors="coerce")
            pearson_r, pearson_p = stats.pearsonr(s1.dropna(), s2.dropna())
            spear_r, spear_p = stats.spearmanr(s1, s2, nan_policy="omit")
            res.update({"pearson_r": pearson_r, "pearson_p": pearson_p,
                        "spearman_r": spear_r, "spearman_p": spear_p})
    except Exception as e:
        res["error"] = str(e)
    return res

def plot_descriptives(df, types):
    figs = []

    # Histograma para variaveis numéricas - Atenção para possiveis erros - reportar para Ferrnando Freua
    for col, t in types.items():
        if t == "numeric":
            fig = plt.figure()
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            plt.hist(s, bins=30)
            plt.title(f"Histograma - {col}")
            plt.xlabel(col); plt.ylabel("Frequência")
            figs.append(("hist_"+col, fig))

    # Barras para categoricas
    for col, t in types.items():
        if t == "categorical":
            fig = plt.figure()
            vc = df[col].astype(str).value_counts().head(20)
            vc.plot(kind="bar")
            plt.title(f"Frequências - {col}")
            plt.xlabel(col); plt.ylabel("Contagem")
            plt.tight_layout()
            figs.append(("bar_"+col, fig))

    # Correlações heatmap (numeric)
    num_cols = [c for c,t in types.items() if t=="numeric"]
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True)
        fig = plt.figure()
        plt.imshow(corr, interpolation="nearest")
        plt.xticks(range(len(num_cols)), num_cols, rotation=90)
        plt.yticks(range(len(num_cols)), num_cols)
        plt.title("Matriz de Correlação (Pearson)")
        plt.colorbar()
        plt.tight_layout()
        figs.append(("corr_matrix", fig))

    return figs

def pdf_report(df, types, desc_summaries, test_results, figs):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Cover page
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        plt.axis("off")
        title = "Relatório Automático de Análise"
        subtitle = f"Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}"
        body = "Desenvolvido por Dr Fernando Freua. © 2025"
        plt.text(0.5, 0.8, title, ha="center", va="center", fontsize=20, weight="bold")
        plt.text(0.5, 0.76, subtitle, ha="center", va="center", fontsize=12)
        plt.text(0.5, 0.1, body, ha="center", va="center", fontsize=9)
        pdf.savefig(fig); plt.close(fig)

        # Stats descritivas para numéricas
        if "numeric" in desc_summaries:
            fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
            plt.axis("off")
            table = desc_summaries["numeric"].round(3)
            plt.title("Estatística Descritiva (Numéricas)")
            tbl = plt.table(cellText=table.values, colLabels=table.columns, rowLabels=table.index,
                            loc="center")
            tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1, 1.2)
            pdf.savefig(fig); plt.close(fig)

        # Frequencia para categoricas
        if "categorical" in desc_summaries:
            for col, vc in desc_summaries["categorical"].items():
                fig = plt.figure(figsize=(11.69, 8.27))
                plt.axis("off")
                top = vc.reset_index()
                top.columns = [col, "contagem"]
                top["contagem"] = top["contagem"].astype(int)
                head = top.head(25)
                plt.title(f"Frequências ({col})")
                tbl = plt.table(cellText=head.values, colLabels=head.columns, loc="center")
                tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.2)
                pdf.savefig(fig); plt.close(fig)

        # Datetime - Sumário geral
        if "datetime" in desc_summaries:
            fig = plt.figure(figsize=(11.69, 8.27))
            plt.axis("off")
            table = desc_summaries["datetime"]
            plt.title("Resumo de Datas")
            tbl = plt.table(cellText=table.values, colLabels=table.columns, rowLabels=table.index,
                            loc="center")
            tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.2)
            pdf.savefig(fig); plt.close(fig)

        # Resultados inferenciais de acordo com validação AI
        if test_results:
            rows = []
            for r in test_results:
                if r.get("error"): 
                    rows.append([r["test"], r["a"], r.get("b",""), "erro", r["error"]])
                else:
                    metric = ""
                    if r["test"]=="t-test": metric = f"t={r['stat']:.3f}  p={r['p_value']:.4g}"
                    if r["test"]=="anova": metric = f"F={r['stat']:.3f} (k={r.get('k_groups',0)})  p={r['p_value']:.4g}"
                    if r["test"]=="chi2": metric = f"chi2={r['chi2']:.3f}  df={r['dof']}  p={r['p_value']:.4g}"
                    if r["test"]=="correlation": metric = f"r(Pearson)={r['pearson_r']:.3f} p={r['pearson_p']:.4g} | ρ(Spearman)={r['spearman_r']:.3f} p={r['spearman_p']:.4g}"
                    rows.append([r["test"], r["a"], r.get("b",""), "ok", metric])
            fig = plt.figure(figsize=(11.69, 8.27))
            plt.axis("off")
            plt.title("Resultados de Testes Inferenciais")
            tbl = plt.table(cellText=rows, colLabels=["Teste","Variável A","Variável B","Status","Resumo"], loc="center")
            tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1, 1.1)
            pdf.savefig(fig); plt.close(fig)

        # Figuras
        for _, fig in figs:
            pdf.savefig(fig); plt.close(fig)

    buf.seek(0)
    return buf

if uploaded:
    try:
        df = load_data(uploaded)
        st.success(f"Arquivo carregado com {df.shape[0]} linhas e {df.shape[1]} colunas.")
        st.dataframe(df.head(50))
        types = detect_types(df)

        st.subheader("📌 Tipos de variáveis (detecção automática)")
        st.json(types)

        st.subheader("📈 Estatística descritiva")
        summaries = summarize_descriptive(df, types)
        if "numeric" in summaries:
            st.markdown("**Numéricas**")
            st.dataframe(summaries["numeric"])
        if "categorical" in summaries:
            st.markdown("**Categóricas (Top 20 por coluna)**")
            for c, vc in summaries["categorical"].items():
                with st.expander(f"Frequências — {c}"):
                    st.write(vc)
        if "datetime" in summaries:
            st.markdown("**Datas**")
            st.dataframe(summaries["datetime"])

        st.subheader("🧠 Sugestões automáticas de análises (IA local)")
        suggestions = suggest_inferential_tests(df, types)
        st.write(pd.DataFrame(suggestions, columns=["teste","variável_a","variável_b"]))

        st.subheader("🧪 Execução dos testes sugeridos")
        results = [run_test(df, *s) for s in suggestions]
        st.write(pd.DataFrame(results))

        st.subheader("📊 Gráficos automáticos")
        figs = plot_descriptives(df, types)
        for name, fig in figs:
            st.pyplot(fig)

        st.subheader("📄 Exportar relatório em PDF")
        if st.button("Gerar PDF"):
            pdf_buf = pdf_report(df, types, summaries, results, figs)
            st.download_button(
                label="Baixar relatório PDF",
                data=pdf_buf,
                file_name="relatorio_automatico.pdf",
                mime="application/pdf",
            )

    except Exception as e:
        st.error(f"Falha ao processar o arquivo: {e}")
else:
    st.info("Envie um arquivo CSV/Excel para começar.")
