import streamlit as st
import pandas as pd
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Macro News RAG Risk Tool",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# LOAD EVERYTHING (cached so it only loads once, not on every query)
# ============================================================
@st.cache_resource
def load_system():
    df = pd.read_pickle('phase3_final.pkl')
    index = faiss.read_index('phase2_index.faiss')
    with open('phase2_data.pkl', 'rb') as f:
        phase2 = pickle.load(f)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return df, index, model

with st.spinner("Loading knowledge base and model (first load takes ~60 seconds)..."):
    df, index, model = load_system()

# ============================================================
# CORE RAG FUNCTIONS
# ============================================================
def query_rag(headline, k=5):
    query_vec = model.encode([headline], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_vec)
    distances, indices = index.search(query_vec, k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), 1):
        row = df.iloc[idx]
        results.append({
            'rank': rank,
            'similarity': round(float(score), 4),
            'date': row['date'].date(),
            'headline': row['headline'][:160],
            'SPY_return_1d': row['SPY_return_1d'],
            'SPY_return_3d': row['SPY_return_3d'],
            'SPY_return_5d': row['SPY_return_5d'],
            'SPY_return_21d': row['SPY_return_21d'],
            'SPY_return_63d': row['SPY_return_63d'],
            'GLD_return_1d': row['GLD_return_1d'],
            'GLD_return_5d': row['GLD_return_5d'],
            'GLD_return_21d': row['GLD_return_21d'],
            'GLD_return_63d': row['GLD_return_63d'],
            'TLT_return_1d': row['TLT_return_1d'],
            'TLT_return_5d': row['TLT_return_5d'],
            'TLT_return_21d': row['TLT_return_21d'],
            'TLT_return_63d': row['TLT_return_63d'],
            'USO_return_1d': row['USO_return_1d'],
            'USO_return_5d': row['USO_return_5d'],
            'USO_return_21d': row['USO_return_21d'],
            'USO_return_63d': row['USO_return_63d'],
        })
    return pd.DataFrame(results)


def weighted_signal(results, asset, horizons):
    weights = results['similarity'].values
    signals = {}
    for h in horizons:
        col = f'{asset}_return_{h}'
        valid = results[col].dropna()
        if len(valid) > 0:
            w = weights[:len(valid)]
            w = w / w.sum()
            signals[h] = np.average(valid.values, weights=w)
        else:
            signals[h] = np.nan
    return signals


def signal_label(value, threshold=0.02):
    if np.isnan(value):
        return "N/A"
    if value < -threshold:
        return "🔴 RISK OFF"
    elif value > threshold:
        return "🟢 RISK ON"
    return "🟡 NEUTRAL"

# ============================================================
# SIDEBAR — ABOUT
# ============================================================
with st.sidebar:
    st.header("About this tool")
    st.write(
        "This is a Retrieval-Augmented Generation (RAG) prototype built for an "
        "academic project. It searches 16+ years of financial news headlines "
        "to find historical analogues to any headline you enter, and shows what "
        "happened next across four asset classes."
    )
    st.markdown("**Knowledge base**")
    st.write(f"{len(df):,} trading days")
    st.write(f"{df['date'].min().date()} → {df['date'].max().date()}")
    st.markdown("**Assets tracked**")
    st.write("SPY (equities) · GLD (gold) · TLT (bonds) · USO (oil)")
    st.markdown("**Known limitations**")
    st.write(
        "- Data gap: April 2024 – May 2026\n"
        "- General-purpose embedding model, not finance-specific\n"
        "- Headline timestamps may lag intraday market events by up to 1 day\n"
        "- This is an academic prototype, not investment advice"
    )

# ============================================================
# MAIN PAGE
# ============================================================
st.title("📊 Macro News → Market Analogue Risk Tool")
st.caption(
    "Enter any financial news headline. The system retrieves the most similar "
    "historical headlines and shows what markets did afterwards."
)

col_input, col_k = st.columns([4, 1])
with col_input:
    headline_input = st.text_input(
        "Headline",
        value="Federal Reserve raises interest rates by 75 basis points surprising markets",
        placeholder="Type or paste a financial news headline..."
    )
with col_k:
    k = st.selectbox("Analogues", [3, 5, 7, 10], index=1)

run = st.button("🔍 Find historical analogues", type="primary")

if run and headline_input.strip():
    with st.spinner("Searching 16 years of financial history..."):
        results = query_rag(headline_input, k=k)

    st.subheader("Top historical analogues")
    display_df = results[['rank', 'date', 'similarity', 'headline']].copy()
    display_df.columns = ['#', 'Date', 'Similarity', 'Headline']
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("Weighted risk signal")
    st.caption("Similarity-weighted average forward return across historical analogues")

    horizons = ['1d', '3d', '5d', '21d', '63d']
    assets = {'SPY': 'Equities', 'GLD': 'Gold', 'TLT': 'Bonds', 'USO': 'Oil'}

    table_rows = []
    for asset, label in assets.items():
        sig = weighted_signal(results, asset, horizons)
        row = {'Asset': f"{asset} ({label})"}
        for h in horizons:
            val = sig[h]
            row[h] = f"{val:.2%}" if not np.isnan(val) else "N/A"
        row['Signal (21d)'] = signal_label(sig['21d'])
        table_rows.append(row)

    signal_df = pd.DataFrame(table_rows)
    st.dataframe(signal_df, use_container_width=True, hide_index=True)

    st.subheader("Detail per analogue")
    for _, row in results.iterrows():
        with st.expander(f"#{row['rank']} [{row['date']}] — similarity {row['similarity']:.4f} — {row['headline'][:90]}"):
            st.write(f"**Headline:** {row['headline']}")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("SPY 1d", f"{row['SPY_return_1d']:.2%}" if pd.notna(row['SPY_return_1d']) else "N/A")
                st.metric("SPY 21d", f"{row['SPY_return_21d']:.2%}" if pd.notna(row['SPY_return_21d']) else "N/A")
            with c2:
                st.metric("GLD 1d", f"{row['GLD_return_1d']:.2%}" if pd.notna(row['GLD_return_1d']) else "N/A")
                st.metric("GLD 21d", f"{row['GLD_return_21d']:.2%}" if pd.notna(row['GLD_return_21d']) else "N/A")
            with c3:
                st.metric("TLT 1d", f"{row['TLT_return_1d']:.2%}" if pd.notna(row['TLT_return_1d']) else "N/A")
                st.metric("TLT 21d", f"{row['TLT_return_21d']:.2%}" if pd.notna(row['TLT_return_21d']) else "N/A")
            with c4:
                st.metric("USO 1d", f"{row['USO_return_1d']:.2%}" if pd.notna(row['USO_return_1d']) else "N/A")
                st.metric("USO 21d", f"{row['USO_return_21d']:.2%}" if pd.notna(row['USO_return_21d']) else "N/A")

    st.info(
        "⚠️ This is an academic prototype for a school project. Outputs are based on "
        "historical pattern matching only and do not constitute financial advice."
    )
elif run:
    st.warning("Please enter a headline first.")
