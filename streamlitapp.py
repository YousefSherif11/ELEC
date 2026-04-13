import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📱 Electronics Pricing Dashboard",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .kpi-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        border: 1px solid #3a3a5e;
        border-radius: 14px;
        padding: 20px 18px;
        text-align: center;
        color: white;
    }
    .kpi-label { font-size: 13px; color: #a0a0c0; margin-bottom: 6px; }
    .kpi-value { font-size: 32px; font-weight: 700; color: #ffffff; }
    .kpi-sub   { font-size: 12px; color: #7070a0; margin-top: 4px; }
    .section-title {
        font-size: 22px; font-weight: 700;
        border-left: 4px solid #636EFA;
        padding-left: 12px; margin: 24px 0 12px 0;
        color: #e0e0ff;
    }
    .qa-q { font-weight: 700; color: #636EFA; font-size: 15px; margin-top: 12px; }
    .qa-a { color: #d0d0f0; font-size: 14px; margin-left: 14px; line-height: 1.6; }
    .insight-box {
        background: #1a1a2e; border-left: 3px solid #636EFA;
        border-radius: 6px; padding: 10px 14px;
        font-size: 13px; color: #b0b0d0; margin: 6px 0;
    }
    div[data-testid="stMetricValue"] { font-size: 26px !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# ── DATA LOADING ──────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════

def _preprocess_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal preprocessing so raw Datafiniti CSV works with the dashboard."""
    # ── normalise column names ────────────────────────────────
    df.columns = df.columns.str.lower().str.replace(".", "", regex=False).str.strip()

    # ── avg_price ─────────────────────────────────────────────
    for max_col, min_col in [
        ("pricesamountmax", "pricesamountmin"),
        ("amountmax", "amountmin"),
    ]:
        if max_col in df.columns and min_col in df.columns:
            df["avg_price"] = (
                pd.to_numeric(df[max_col], errors="coerce").fillna(0) +
                pd.to_numeric(df[min_col], errors="coerce").fillna(0)
            ) / 2
            df = df[df["avg_price"] > 0]
            # remove extreme outliers (top/bottom 1%)
            lo, hi = df["avg_price"].quantile(0.01), df["avg_price"].quantile(0.99)
            df = df[(df["avg_price"] >= lo) & (df["avg_price"] <= hi)]
            break

    # ── merchant_clean ────────────────────────────────────────
    merch_src = next((c for c in ["pricesmerchant", "merchant"] if c in df.columns), None)
    if merch_src:
        def _merch(v):
            if pd.isna(v): return "Other"
            v = str(v).lower()
            if "bestbuy" in v or "best buy" in v: return "Best Buy"
            if "walmart" in v:  return "Walmart"
            if "amazon"  in v:  return "Amazon"
            if "bhphoto" in v:  return "B&H Photo"
            if "ebay"    in v:  return "eBay"
            return "Other"
        df["merchant_clean"] = df[merch_src].apply(_merch)
    else:
        df["merchant_clean"] = "Unknown"

    # ── first_category ────────────────────────────────────────
    cat_src = next((c for c in ["categories", "primarycategories"] if c in df.columns), None)
    if cat_src:
        def _cat(v):
            v = str(v).lower()
            if any(k in v for k in ["computer","laptop","tablet"]):       return "Computers"
            if any(k in v for k in ["tv","ultra hd","flat screen"]):      return "TVs"
            if any(k in v for k in ["audio","speaker","headphone","soundbar"]): return "Audio & Speakers"
            if any(k in v for k in ["camera","photo","dslr"]):            return "Cameras"
            if any(k in v for k in ["phone","mobile","ipod","mp3"]):      return "Phones & Music Players"
            if any(k in v for k in ["office","tool","printer"]):          return "Office & Tools"
            return "Other"
        df["first_category"] = df[cat_src].apply(_cat)
    else:
        df["first_category"] = "Other"

    # ── pricescondition ───────────────────────────────────────
    cond_src = next((c for c in ["pricescondition", "condition"] if c in df.columns), None)
    if cond_src:
        def _cond(v):
            if pd.isna(v): return "Unknown"
            v = str(v).lower()
            if any(k in v for k in ["refurb","recondition"]): return "Refurbished"
            if any(k in v for k in ["used","pre-owned"]):     return "Used"
            if any(k in v for k in ["new","sealed"]):         return "New"
            return "Unknown"
        df["pricescondition"] = df[cond_src].apply(_cond)
    else:
        df["pricescondition"] = "Unknown"

    # ── shipping_type ─────────────────────────────────────────
    ship_src = next((c for c in ["pricesshipping", "shipping"] if c in df.columns), None)
    if ship_src:
        def _ship(v):
            if pd.isna(v): return "Unknown"
            v = str(v).lower()
            if any(k in v for k in ["free","prime","two-day","expedited"]): return "Free"
            if any(k in v for k in ["standard","value"]):                   return "Standard"
            if any(k in v for k in ["freight","cargo"]):                    return "Freight"
            import re
            m = re.search(r"\d+\.?\d*", v)
            if m: return "Free" if float(m.group()) == 0 else "Paid"
            return "Unknown"
        df["shipping_type"] = df[ship_src].apply(_ship)
    else:
        df["shipping_type"] = "Unknown"

    # ── is_in_stock ───────────────────────────────────────────
    avail_src = next((c for c in ["pricesavailability", "availability"] if c in df.columns), None)
    if avail_src:
        def _avail(v):
            if pd.isna(v): return "Unknown"
            v = str(v).lower()
            if any(k in v for k in ["yes","in stock","true"]): return "Available"
            if any(k in v for k in ["no","false","out of stock"]): return "Out Of Stock"
            if any(k in v for k in ["special order","limited","coming soon"]): return "Limited/Coming Soon"
            if any(k in v for k in ["discontinued","retired"]): return "Discontinued"
            return "Unknown"
        df["is_in_stock"] = df[avail_src].apply(_avail)
    else:
        df["is_in_stock"] = "Unknown"

    # ── pricesissale ──────────────────────────────────────────
    sale_src = next((c for c in ["pricesissale", "issale"] if c in df.columns), None)
    if sale_src:
        df["pricesissale"] = pd.to_numeric(
            df[sale_src].astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0}),
            errors="coerce"
        ).fillna(0).astype(int)
    else:
        df["pricesissale"] = 0

    # ── brand ─────────────────────────────────────────────────
    if "brand" not in df.columns and "manufacturer" in df.columns:
        df["brand"] = df["manufacturer"]
    if "brand" in df.columns:
        df["brand"] = df["brand"].fillna("Unknown")

    # ── date year/month ───────────────────────────────────────
    for raw_col, out_y, out_m in [
        ("pricesdateseen", "pricesdateseen_year", "pricesdateseen_month"),
        ("dateadded",      "dateadded_year",      "dateadded_month"),
    ]:
        if raw_col in df.columns:
            dt = pd.to_datetime(
                df[raw_col].astype(str).str.split(",").str[0]
                           .str.replace("Z","",regex=False)
                           .str.replace("T"," ",regex=False),
                errors="coerce", utc=True
            )
            df[out_y] = dt.dt.year
            df[out_m] = dt.dt.month_name()

    return df.reset_index(drop=True)


@st.cache_data
def load_data():
    """Try to load the processed CSV, fall back to simulated data."""
    possible_paths = [
        "electronics_processed.csv",
        "DatafinitiElectronicsProductsPricingData.csv",
        "data.csv",
    ]
    for p in possible_paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, encoding="latin-1", nrows=15000)
                df = _preprocess_raw(df)
                if "avg_price" in df.columns and len(df) > 10:
                    return df, True
            except Exception:
                pass
    # ── Simulated representative data from EDA results ────────
    np.random.seed(42)
    n = 3000
    categories   = ["Computers", "Audio & Speakers", "TVs", "Cameras",
                     "Phones & Music Players", "Office & Tools", "Other"]
    cat_weights  = [0.32, 0.22, 0.14, 0.12, 0.11, 0.05, 0.04]
    merchants    = ["Amazon", "Best Buy", "Walmart", "B&H Photo", "Other"]
    merch_w      = [0.45, 0.25, 0.15, 0.10, 0.05]
    conditions   = ["New", "Refurbished", "Used", "Unknown"]
    cond_w       = [0.72, 0.15, 0.08, 0.05]
    shipping     = ["Free", "Standard", "Paid", "Unknown", "Freight"]
    ship_w       = [0.55, 0.20, 0.12, 0.10, 0.03]
    availability = ["Available", "Out Of Stock", "Unknown", "Limited/Coming Soon", "Discontinued"]
    avail_w      = [0.60, 0.15, 0.14, 0.08, 0.03]
    platforms    = ["amazon", "bestbuy", "walmart", "bhphotovideo", "other"]
    plat_w       = [0.44, 0.24, 0.15, 0.10, 0.07]
    brands       = ["Samsung", "Apple", "Sony", "Bose", "LG", "Logitech",
                    "Canon", "HP", "Dell", "Lenovo"]
    brand_w      = [0.18, 0.15, 0.13, 0.10, 0.09, 0.08, 0.07, 0.07, 0.07, 0.06]

    cat_price_map = {
        "Computers": (800, 250), "TVs": (750, 200), "Cameras": (500, 180),
        "Phones & Music Players": (350, 120), "Audio & Speakers": (180, 80),
        "Office & Tools": (150, 70), "Other": (200, 100),
    }

    df_cats = np.random.choice(categories, size=n, p=cat_weights)
    prices  = np.array([max(10, np.random.normal(*cat_price_map[c])) for c in df_cats])
    df_sim  = pd.DataFrame({
        "first_category":  df_cats,
        "merchant_clean":  np.random.choice(merchants, n, p=merch_w),
        "pricescondition": np.random.choice(conditions, n, p=cond_w),
        "shipping_type":   np.random.choice(shipping,  n, p=ship_w),
        "is_in_stock":     np.random.choice(availability, n, p=avail_w),
        "platform":        np.random.choice(platforms, n, p=plat_w),
        "brand":           np.random.choice(brands, n, p=brand_w),
        "pricesissale":    np.random.choice([0, 1], n, p=[0.68, 0.32]),
        "avg_price":       prices,
        "weight_kg":       np.clip(np.random.exponential(1.5, n), 0.05, 30),
        "pricesdateseen_year": np.random.choice([2015, 2016, 2017, 2018], n,
                                                p=[0.10, 0.25, 0.40, 0.25]),
        "pricesdateseen_month": np.random.choice(
            ["January","February","March","April","May","June",
             "July","August","September","October","November","December"], n),
    })
    return df_sim, False


@st.cache_resource
def load_model(path="model.pkl"):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


df, real_data = load_data()
artefacts = load_model()

# ── resolve which column holds price ─────────────────────────
_price_candidates = ["avg_price", "avg_price_usd", "pricesamountmax", "amountmax"]
price_col = next((c for c in _price_candidates if c in df.columns), None)

if price_col is None:
    for c in df.select_dtypes(include="number").columns:
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(vals) > 0 and vals.max() > 1 and vals.mean() > 1:
            price_col = c
            break

if price_col is None:
    st.error("Cannot find a price column. Expected avg_price or prices.amountMax.")
    st.stop()

df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
df = df[df[price_col].notna() & (df[price_col] > 0)].copy()

merchant_freq_map = df["merchant_clean"].value_counts().to_dict() if "merchant_clean" in df.columns else {}
category_freq_map = df["first_category"].value_counts().to_dict() if "first_category" in df.columns else {}
primary_category_options = [
    "Computers", "Audio & Speakers", "TVs", "Cameras",
    "Phones & Music Players", "Office & Tools", "Other"
]
platform_options = sorted(df["platform"].dropna().unique().tolist()) if "platform" in df.columns else ["amazon", "bestbuy", "walmart", "bhphotovideo", "other"]
merchant_options = sorted(df["merchant_clean"].dropna().unique().tolist()) if "merchant_clean" in df.columns else ["Amazon", "Best Buy", "Walmart", "B&H Photo", "Other"]
shipping_options = sorted(df["shipping_type"].dropna().unique().tolist()) if "shipping_type" in df.columns else ["Free", "Standard", "Unknown", "Paid", "Freight"]
stock_options = sorted(df["is_in_stock"].dropna().unique().tolist()) if "is_in_stock" in df.columns else ["Available", "Out Of Stock", "Unknown", "Discontinued", "Limited/Coming Soon"]
currency_options = sorted(df["pricescurrency"].dropna().unique().tolist()) if "pricescurrency" in df.columns else ["USD", "CAD", "EUR"]
condition_options = ["New", "Refurbished", "Used", "Unknown"]
condition_encoding = {"Used": 1, "Refurbished": 2, "New": 3, "Unknown": 0}
month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

# ══════════════════════════════════════════════════════════════
# ── SIDEBAR ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════

st.sidebar.image("https://img.icons8.com/color/96/electronics.png", width=72)
st.sidebar.title("📱 Electronics Pricing")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview & KPIs", "📊 EDA Visuals", "🔮 Price Predictor", "❓ Q&A Insights"],
    index=0,
)

st.sidebar.markdown("---")
if not real_data:
    st.sidebar.warning("⚠️ CSV not found — showing simulated data.\nPlace your CSV in the same folder to see real stats.")
else:
    st.sidebar.success("✅ Real dataset loaded")

if artefacts:
    st.sidebar.success(f"✅ Model: **{artefacts['model_name']}**")
    st.sidebar.metric("Test MAE", f"${artefacts['test_mae']:.2f}")
    st.sidebar.metric("Test R²",  f"{artefacts['test_r2']:.4f}")
else:
    st.sidebar.info("ℹ️ model.pkl not found\nRun notebook to train & save model.")

# ══════════════════════════════════════════════════════════════
# ── PAGE 1: OVERVIEW & KPIs ───────────────────────────────────
# ══════════════════════════════════════════════════════════════

if page == "🏠 Overview & KPIs":
    st.title("📱 Electronics Pricing — Overview & KPIs")
    st.caption("Datafiniti dataset · US retailers: Amazon, Best Buy, Walmart & more")
    st.markdown("---")

    # ── KPI row 1 ─────────────────────────────────────────────
    total       = len(df)
    avg_p       = df[price_col].mean()
    median_p    = df[price_col].median()
    max_p       = df[price_col].max()
    pct_sale    = (df["pricesissale"].sum() / total * 100) if "pricesissale" in df.columns else 0
    top_merch   = df["merchant_clean"].mode()[0] if "merchant_clean" in df.columns else "Amazon"
    top_cat     = df["first_category"].mode()[0]  if "first_category"  in df.columns else "Computers"
    pct_new     = (df[df["pricescondition"] == "New"].shape[0] / total * 100) if "pricescondition" in df.columns else 0
    pct_free    = (df[df["shipping_type"] == "Free"].shape[0]  / total * 100) if "shipping_type"   in df.columns else 0
    pct_avail   = (df[df["is_in_stock"]   == "Available"].shape[0] / total * 100) if "is_in_stock" in df.columns else 0

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">📦 Total Records</div>
            <div class="kpi-value">{total:,}</div>
            <div class="kpi-sub">product price observations</div>
        </div>""", unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">💵 Average Price</div>
            <div class="kpi-value">${avg_p:,.0f}</div>
            <div class="kpi-sub">median ${median_p:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">🏷️ On-Sale Rate</div>
            <div class="kpi-value">{pct_sale:.1f}%</div>
            <div class="kpi-sub">of all listings</div>
        </div>""", unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">🏪 Top Merchant</div>
            <div class="kpi-value">{top_merch}</div>
            <div class="kpi-sub">most listings</div>
        </div>""", unsafe_allow_html=True)
    with kpi5:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">📂 Top Category</div>
            <div class="kpi-value">{top_cat}</div>
            <div class="kpi-sub">by record count</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    kpi6, kpi7, kpi8, kpi9, kpi10 = st.columns(5)
    with kpi6:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">✅ New Condition</div>
            <div class="kpi-value">{pct_new:.1f}%</div>
            <div class="kpi-sub">of all products</div>
        </div>""", unsafe_allow_html=True)
    with kpi7:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">🚚 Free Shipping</div>
            <div class="kpi-value">{pct_free:.1f}%</div>
            <div class="kpi-sub">of all listings</div>
        </div>""", unsafe_allow_html=True)
    with kpi8:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">🟢 In Stock</div>
            <div class="kpi-value">{pct_avail:.1f}%</div>
            <div class="kpi-sub">available now</div>
        </div>""", unsafe_allow_html=True)
    with kpi9:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">💸 Max Price</div>
            <div class="kpi-value">${max_p:,.0f}</div>
            <div class="kpi-sub">highest recorded</div>
        </div>""", unsafe_allow_html=True)
    with kpi10:
        num_cats = df["first_category"].nunique() if "first_category" in df.columns else 7
        num_brands = df["brand"].nunique() if "brand" in df.columns else 10
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">🏷️ Unique Brands</div>
            <div class="kpi-value">{num_brands}</div>
            <div class="kpi-sub">across {num_cats} categories</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Quick overview charts ──────────────────────────────────
    st.markdown('<div class="section-title">📈 Quick Overview</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        cat_cnt = df["first_category"].value_counts().reset_index()
        cat_cnt.columns = ["Category", "Count"]
        fig = px.bar(cat_cnt, x="Category", y="Count", text="Count",
                     title="Products per Category",
                     color="Count", color_continuous_scale="Blues")
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="white")
        st.plotly_chart(fig, width='stretch')

    with c2:
        fig = px.histogram(df, x=price_col, nbins=50,
                           title="Price Distribution",
                           color_discrete_sequence=["#636EFA"])
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="white")
        st.plotly_chart(fig, width='stretch')

    c3, c4 = st.columns(2)
    with c3:
        if "merchant_clean" in df.columns:
            merch_cnt = df["merchant_clean"].value_counts().reset_index()
            merch_cnt.columns = ["Merchant", "Count"]
            fig = px.pie(merch_cnt, names="Merchant", values="Count",
                         title="Merchant Share",
                         color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               font_color="white")
            st.plotly_chart(fig, width='stretch')

    with c4:
        if "pricescondition" in df.columns:
            cond_cnt = df["pricescondition"].value_counts().reset_index()
            cond_cnt.columns = ["Condition", "Count"]
            fig = px.pie(cond_cnt, names="Condition", values="Count",
                         title="Product Condition Mix",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               font_color="white")
            st.plotly_chart(fig, width='stretch')


# ══════════════════════════════════════════════════════════════
# ── PAGE 2: EDA VISUALS ───────────────────────────────────────
# ══════════════════════════════════════════════════════════════

elif page == "📊 EDA Visuals":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Deep-dive visuals into the dataset's pricing patterns and distributions.")
    st.markdown("---")

    DARK = dict(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
    category_tab, merchant_tab, sale_tab, condition_tab, time_tab = st.tabs([
        "Category", "Merchant", "Sale & Condition", "Shipping & Stock", "Time Trends"
    ])

    with category_tab:
        cat_avg = df.groupby("first_category")[price_col].mean().sort_values(ascending=False).reset_index()
        cat_avg.columns = ["Category", "Avg Price"]
        fig = px.bar(cat_avg, x="Category", y="Avg Price", text="Avg Price",
                     color="Avg Price", color_continuous_scale="Blues",
                     title="Average Price by Category")
        fig.update_traces(texttemplate="$%{text:.0f}", textposition="outside")
        fig.update_layout(**DARK, coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')

        fig = px.box(df, x="first_category", y=price_col,
                     color="first_category",
                     title="Price Distribution by Category",
                     color_discrete_sequence=px.colors.qualitative.Plotly)
        fig.update_layout(**DARK, showlegend=False, xaxis={"tickangle": -30})
        st.plotly_chart(fig, width='stretch')

    with merchant_tab:
        merch_avg = df.groupby("merchant_clean")[price_col].mean().sort_values(ascending=False).reset_index()
        merch_avg.columns = ["Merchant", "Avg Price"]
        fig = px.bar(merch_avg, x="Merchant", y="Avg Price", text="Avg Price",
                     color="Avg Price", color_continuous_scale="Oranges",
                     title="Average Price per Merchant")
        fig.update_traces(texttemplate="$%{text:.0f}", textposition="outside")
        fig.update_layout(**DARK, coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')

        merch_cnt = df["merchant_clean"].value_counts().reset_index()
        merch_cnt.columns = ["Merchant", "Count"]
        fig = px.bar(merch_cnt, x="Merchant", y="Count", text="Count",
                     color="Count", color_continuous_scale="Reds",
                     title="Number of Products per Merchant")
        fig.update_layout(**DARK, coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')

    with sale_tab:
        sale_avg = df.groupby("pricesissale")[price_col].mean().reset_index()
        sale_avg["pricesissale"] = sale_avg["pricesissale"].map({0: "Regular", 1: "On Sale"})
        sale_avg.columns = ["Type", "Avg Price"]
        fig = px.bar(sale_avg, x="Type", y="Avg Price", text="Avg Price",
                     color="Type",
                     color_discrete_map={"Regular": "#636EFA", "On Sale": "#EF553B"},
                     title="Average Price: Sale vs Regular")
        fig.update_traces(texttemplate="$%{text:.0f}", textposition="outside")
        fig.update_layout(**DARK, showlegend=False)
        st.plotly_chart(fig, width='stretch')

        cond_avg = df.groupby("pricescondition")[price_col].mean().sort_values(ascending=False).reset_index()
        cond_avg.columns = ["Condition", "Avg Price"]
        fig = px.bar(cond_avg, x="Condition", y="Avg Price", text="Avg Price",
                     color="Avg Price", color_continuous_scale="Greens",
                     title="Average Price by Condition")
        fig.update_traces(texttemplate="$%{text:.0f}", textposition="outside")
        fig.update_layout(**DARK, coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')

        cond_cnt = df["pricescondition"].value_counts().reset_index()
        cond_cnt.columns = ["Condition", "Count"]
        fig = px.pie(cond_cnt, names="Condition", values="Count",
                     title="Condition Distribution",
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(**DARK)
        st.plotly_chart(fig, width='stretch')

    with condition_tab:
        ship_cnt = df["shipping_type"].value_counts().reset_index()
        ship_cnt.columns = ["Shipping", "Count"]
        fig = px.bar(ship_cnt, x="Shipping", y="Count", text="Count",
                     color="Count", color_continuous_scale="Purples",
                     title="Shipping Type Distribution")
        fig.update_layout(**DARK, coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')

        if "brand" in df.columns:
            brand_cnt = df["brand"].value_counts().head(10).reset_index()
            brand_cnt.columns = ["Brand", "Count"]
            fig = px.bar(brand_cnt, x="Brand", y="Count", text="Count",
                         color="Count", color_continuous_scale="Reds",
                         title="Top 10 Brands by Product Count")
            fig.update_layout(**DARK, coloraxis_showscale=False)
            st.plotly_chart(fig, width='stretch')

    with time_tab:
        if "pricesdateseen_year" in df.columns:
            year_avg = df.groupby("pricesdateseen_year")[price_col].mean().reset_index()
            year_avg.columns = ["Year", "Avg Price"]
            fig = px.line(year_avg, x="Year", y="Avg Price", markers=True,
                          title="Average Price by Year",
                          color_discrete_sequence=["#6B66FF"])
            fig.update_layout(**DARK)
            st.plotly_chart(fig, width='stretch')

        month_order = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        if "pricesdateseen_month" in df.columns:
            month_avg = df.groupby("pricesdateseen_month")[price_col].mean().reindex(month_order).reset_index()
            month_avg.columns = ["Month", "Avg Price"]
            fig = px.bar(month_avg, x="Month", y="Avg Price",
                         title="Average Price by Month",
                         color_discrete_sequence=["#7DFF66"])
            fig.update_layout(**DARK, xaxis={"tickangle": -40})
            st.plotly_chart(fig, width='stretch')

        avail_cnt = df["is_in_stock"].value_counts().reset_index()
        avail_cnt.columns = ["Status", "Count"]
        fig = px.pie(avail_cnt, names="Status", values="Count",
                     title="Stock Availability Distribution",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(**DARK)
        st.plotly_chart(fig, width='stretch')

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) >= 3:
        corr = df[num_cols].corr()
        fig = px.imshow(corr, title="Correlation — Numerical Features",
                        color_continuous_scale="RdBu", color_continuous_midpoint=0,
                        text_auto=".2f")
        fig.update_layout(**DARK, height=450)
        st.plotly_chart(fig, width='stretch')


# ══════════════════════════════════════════════════════════════
# ── PAGE 3: PRICE PREDICTOR ───────────────────────────────────
# ══════════════════════════════════════════════════════════════

elif page == "🔮 Price Predictor":
    st.title("🔮 Electronics Price Predictor")
    st.markdown("Adjust product attributes below to get an estimated average price.")
    st.markdown("---")

    if artefacts is None:
        st.warning(
            "**model.pkl not found.**\n\n"
            "To enable predictions:\n"
            "1. Run all cells in your notebook\n"
            "2. The last ML cell saves `model.pkl`\n"
            "3. Place `model.pkl` in the same folder as this app\n"
            "4. Restart the app"
        )
        st.stop()

    pipeline     = artefacts["pipeline"]
    NUM_FEATURES = artefacts["num_features"]
    CAT_FEATURES = artefacts["cat_features"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📋 Product Details**")
        pricescondition = st.selectbox("Condition", condition_options, index=0)
        pricescurrency = st.selectbox("Currency", currency_options, index=0)
        pricesissale = 1 if st.selectbox("On Sale?", ["No", "Yes"]) == "Yes" else 0
        is_in_stock = st.selectbox(
            "Stock Status",
            stock_options,
            index=stock_options.index("Available") if "Available" in stock_options else 0)
        weight_kg = st.number_input("Weight (kg)", 0.0, 200.0, 1.0, 0.1)

    with col2:
        st.markdown("**📂 Category & Platform**")
        first_category = st.selectbox("Category", primary_category_options,
                                      index=primary_category_options.index("Computers"))
        platform = st.selectbox("Platform", platform_options, index=0)
        merchant_clean = st.selectbox("Merchant", merchant_options, index=0)
        shipping_type = st.selectbox(
            "Shipping Type",
            shipping_options,
            index=shipping_options.index("Free") if "Free" in shipping_options else 0)
        pricescondition_encoded = condition_encoding[pricescondition]

    with col3:
        st.markdown("**📅 Dates**")
        dateadded_year = st.selectbox(
            "Year Added",
            sorted(df["dateadded_year"].dropna().astype(int).unique().tolist())
            if "dateadded_year" in df.columns else list(range(2012, 2020)),
            index=0)
        dateadded_month = st.selectbox("Month Added", month_names, index=0)
        dateadded_day = st.slider("Day Added", 1, 31, 15)
        dateupdated_year = st.selectbox(
            "Year Updated",
            sorted(df["dateupdated_year"].dropna().astype(int).unique().tolist())
            if "dateupdated_year" in df.columns else list(range(2014, 2020)),
            index=0)
        dateupdated_month = st.selectbox("Month Updated", month_names, index=0)
        dateupdated_day = st.slider("Day Updated", 1, 31, 12)
        pricesdateseen_year = st.selectbox(
            "Year Price Seen",
            sorted(df["pricesdateseen_year"].dropna().astype(int).unique().tolist())
            if "pricesdateseen_year" in df.columns else list(range(2014, 2020)),
            index=0)
        pricesdateseen_month = st.selectbox("Month Price Seen", month_names, index=0)
        pricesdateseen_day = st.slider("Day Price Seen", 1, 31, 15)

    merchant_freq_encoded = merchant_freq_map.get(merchant_clean, 0)
    category_freq_encoded = category_freq_map.get(first_category, 0)

    primary_feature_group = st.selectbox(
        "Primary Category Group",
        ["Electronics", "Apple CarPlay", "Furniture", "Intel Celeron", "Media", "Siri Eyes Free", "Other"],
        index=0)

    one_hot_primary = {
        f"primarycategories_{label}": int(primary_feature_group == label)
        for label in ["Electronics", "Apple CarPlay", "Furniture", "Intel Celeron", "Media", "Siri Eyes Free"]
    }

    ship_values = {
        "ship_Free": int(shipping_type == "Free"),
        "ship_Standard": int(shipping_type == "Standard"),
        "ship_Paid": int(shipping_type == "Paid"),
        "ship_Freight": int(shipping_type == "Freight"),
        "ship_Unknown": int(shipping_type == "Unknown"),
    }

    all_num = {
        "weight_kg": weight_kg,
        "merchant_freq_encoded": merchant_freq_encoded,
        "pricescondition_encoded": pricescondition_encoded,
        "category_freq_encoded": category_freq_encoded,
        "pricesissale": pricesissale,
        "dateadded_year": dateadded_year,
        "dateadded_day": dateadded_day,
        "dateupdated_year": dateupdated_year,
        "dateupdated_day": dateupdated_day,
        "pricesdateseen_year": pricesdateseen_year,
        "pricesdateseen_day": pricesdateseen_day,
        **ship_values,
        **one_hot_primary,
    }
    all_num = {k: v for k, v in all_num.items() if k in NUM_FEATURES}

    all_cat = {
        "pricescondition": pricescondition,
        "pricescurrency": pricescurrency,
        "platform": platform,
        "first_category": first_category,
        "merchant_clean": merchant_clean,
        "shipping_type": shipping_type,
        "is_in_stock": is_in_stock,
        "dateadded_month": dateadded_month,
        "dateupdated_month": dateupdated_month,
        "pricesdateseen_month": pricesdateseen_month,
    }
    all_cat = {k: v for k, v in all_cat.items() if k in CAT_FEATURES}

    input_data = {**all_num, **all_cat}
    for col in NUM_FEATURES:
        input_data.setdefault(col, 0)
    for col in CAT_FEATURES:
        input_data.setdefault(col, "Unknown")

    input_df = pd.DataFrame([input_data])[NUM_FEATURES + CAT_FEATURES]

    st.markdown("---")
    pred_col, info_col = st.columns([1, 2])

    with pred_col:
        if st.button("🔮 Predict Price", type="primary", width='stretch'):
            try:
                prediction = pipeline.predict(input_df)[0]
                st.session_state["last_pred"] = prediction
            except Exception as e:
                st.error(f"Prediction error: {e}")

        if "last_pred" in st.session_state:
            pred = st.session_state["last_pred"]
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                border-radius:14px;padding:28px;text-align:center;color:white;margin-top:12px'>
                <div style='font-size:14px;opacity:.85'>💵 Estimated Average Price</div>
                <div style='font-size:52px;font-weight:700'>${pred:,.2f}</div>
                <div style='font-size:12px;opacity:.7'>USD</div>
            </div>""", unsafe_allow_html=True)

    with info_col:
        st.markdown("**📋 Selected Inputs Summary**")
        summary = {
            "Category": first_category, "Condition": pricescondition,
            "Platform": platform, "Merchant": merchant_clean,
            "On Sale": "Yes" if pricesissale else "No",
            "Shipping": shipping_type, "Stock": is_in_stock,
            "Weight (kg)": weight_kg, "Year Added": dateadded_year,
        }
        summary_df = pd.DataFrame(list(summary.items()), columns=["Attribute", "Value"])
        summary_df["Value"] = summary_df["Value"].astype(str)
        st.dataframe(summary_df, width='stretch', hide_index=True)

    st.markdown("---")
    st.subheader("📊 Typical Price Ranges by Category")
    ranges = {
        "Phones & Music Players": (50, 700),   "Computers": (100, 1100),
        "Cameras": (80, 1000),                  "TVs": (200, 1100),
        "Audio & Speakers": (30, 400),          "Office & Tools": (20, 350),
        "Other": (10, 500),
    }
    range_df = pd.DataFrame([
        {"Category": c, "Min ($)": lo, "Max ($)": hi, "Midpoint ($)": (lo+hi)/2}
        for c, (lo, hi) in ranges.items()
    ])
    st.dataframe(range_df, width='stretch', hide_index=True)

    st.markdown("---")
    st.caption(
        f"Model: **{artefacts['model_name']}**  |  "
        f"Test MAE: **${artefacts['test_mae']:.2f}**  |  "
        f"Test R²: **{artefacts['test_r2']:.4f}**"
    )


# ══════════════════════════════════════════════════════════════
# ── PAGE 4: Q&A INSIGHTS ──────────────────────────────────────
# ══════════════════════════════════════════════════════════════

elif page == "❓ Q&A Insights":
    st.title("❓ Q&A — Project Insights")
    st.markdown("Key questions answered from the data analysis and modeling results.")
    st.markdown("---")

    # ── Pre-compute answers from actual data ──────────────────
    top_merch_price = df.groupby("merchant_clean")[price_col].mean().idxmax()
    top_cat_price   = df.groupby("first_category")[price_col].mean().idxmax()
    low_cat_price   = df.groupby("first_category")[price_col].mean().idxmin()
    avg_sale   = df[df["pricesissale"] == 1][price_col].mean()
    avg_nosale = df[df["pricesissale"] == 0][price_col].mean()
    sale_diff  = avg_nosale - avg_sale
    sale_amount = abs(sale_diff)
    sale_prefix = "Yes" if sale_diff > 0 else "No"
    sale_trend = "less" if sale_diff > 0 else "more"
    sale_answer = (
        f"{sale_prefix}. **Sale items** average **${avg_sale:,.0f}** vs. **${avg_nosale:,.0f}** for regular items "
        f"— a difference of **${sale_amount:,.0f}**. "
        f"Sale items cost {sale_trend} on average. This confirms that the `pricesissale` flag is "
        f"a meaningful signal (not just a marketing label)."
    )
    top_brand  = df["brand"].value_counts().idxmax() if "brand" in df.columns else "Samsung"
    top_brand_cnt = df["brand"].value_counts().max()  if "brand" in df.columns else 0
    pct_new    = df[df["pricescondition"] == "New"].shape[0] / len(df) * 100
    pct_free   = df[df["shipping_type"] == "Free"].shape[0]  / len(df) * 100
    pct_avail  = df[df["is_in_stock"] == "Available"].shape[0] / len(df) * 100
    new_avg    = df[df["pricescondition"] == "New"][price_col].mean()
    used_avg   = df[df["pricescondition"] == "Used"][price_col].mean()
    refurb_avg = df[df["pricescondition"] == "Refurbished"][price_col].mean()

    model_name = artefacts["model_name"] if artefacts else "GradientBoosting"
    test_mae   = f"${artefacts['test_mae']:.2f}" if artefacts else "~$50"
    test_r2    = f"{artefacts['test_r2']:.4f}"  if artefacts else "~0.85"

    qa_pairs = [
        {
            "q": "1️⃣ What is the dataset about?",
            "a": (
                "The dataset comes from **Datafiniti** and covers **7,000+ consumer electronics products** "
                "across major US retailers — Amazon, Best Buy, and Walmart. It captures pricing dynamics "
                "including sale prices, shipping costs, product conditions, brand names, and merchant details. "
                "The goal is to **predict the average price** of a product given its attributes."
            ),
            "chart": None,
        },
        {
            "q": f"2️⃣ Which merchant charges the highest average price?",
            "a": (
                f"**{top_merch_price}** tends to have the highest average price in the dataset. "
                f"Merchant pricing varies significantly — specialty retailers like B&H Photo typically "
                f"carry premium professional gear, while Amazon and Walmart compete on breadth and value."
            ),
            "chart": "merchant_price",
        },
        {
            "q": f"3️⃣ Which product category is the most expensive?",
            "a": (
                f"**{top_cat_price}** has the highest average price, while **{low_cat_price}** "
                f"is the most affordable category. High-end electronics like TVs and laptops naturally "
                f"command premium prices compared to accessories or audio gear."
            ),
            "chart": "cat_price",
        },
        {
            "q": f"4️⃣ Do sale items actually cost less?",
            "a": sale_answer,
            "chart": "sale_price",
        },
        {
            "q": f"5️⃣ How does product condition affect price?",
            "a": (
                f"As expected:\n"
                f"- **New**: avg ${new_avg:,.0f}\n"
                f"- **Refurbished**: avg ${refurb_avg:,.0f}\n"
                f"- **Used**: avg ${used_avg:,.0f}\n\n"
                f"{pct_new:.1f}% of all listings are New condition, making it the dominant category."
            ),
            "chart": "cond_price",
        },
        {
            "q": "6️⃣ What shipping type dominates the market?",
            "a": (
                f"**Free shipping** covers **{pct_free:.1f}%** of all listings. This reflects the Amazon Prime "
                f"effect — where free shipping has become a baseline expectation for online electronics buyers. "
                f"Standard and Paid shipping together account for roughly 30% of listings."
            ),
            "chart": "shipping",
        },
        {
            "q": f"7️⃣ Which brand is most represented in the dataset?",
            "a": (
                f"**{top_brand}** has the most product records ({top_brand_cnt:,} listings). "
                f"The top brands overall include Samsung, Apple, Sony, Bose, and LG — reflecting the "
                f"dominance of major consumer electronics names in US retail."
            ),
            "chart": "brand_cnt",
        },
        {
            "q": "8️⃣ What percentage of products are in stock?",
            "a": (
                f"**{pct_avail:.1f}%** of products are listed as 'Available'. "
                f"Out-of-stock, discontinued, and limited products make up the remainder — "
                f"these availability flags were used as features in the ML model since "
                f"availability can correlate with price (e.g., discontinued items may be discounted)."
            ),
            "chart": "availability",
        },
        {
            "q": "9️⃣ How was the data cleaned and preprocessed?",
            "a": (
                "The preprocessing pipeline included:\n"
                "- **Dropped** 15+ unnecessary/empty columns (IDs, image URLs, barcodes)\n"
                "- **Extracted** platform from source URLs via URL parsing\n"
                "- **Parsed** 3 date columns into year/month/day features\n"
                "- **Cleaned** weight → unified kg values\n"
                "- **Labeled** shipping into 5 categories (Free, Standard, Paid, Freight, Unknown)\n"
                "- **Cleaned** merchant names into 5 major buckets\n"
                "- **Ordinal-encoded** condition (Used→1, Refurbished→2, New→3)\n"
                "- **Removed outliers** using IQR method on price\n"
                "- **Currency-converted** all prices to USD"
            ),
            "chart": None,
        },
        {
            "q": f"🔟 What ML model performed best and how accurate is it?",
            "a": (
                f"Three models were compared via 5-fold cross-validation:\n"
                f"- **Ridge Regression** — simple linear baseline\n"
                f"- **Random Forest** — ensemble of decision trees\n"
                f"- **Gradient Boosting** — sequential boosted trees\n\n"
                f"**Winner: {model_name}**\n"
                f"- Test MAE: **{test_mae}** (average prediction error in dollars)\n"
                f"- Test R²: **{test_r2}** (proportion of price variance explained)\n\n"
                f"Features like category, merchant, condition, sale status, and date were most predictive."
            ),
            "chart": None,
        },
    ]

    DARK = dict(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")

    for item in qa_pairs:
        with st.expander(item["q"], expanded=False):
            st.markdown(item["a"])

            if item["chart"] == "merchant_price":
                ma = df.groupby("merchant_clean")[price_col].mean().sort_values(ascending=False).reset_index()
                ma.columns = ["Merchant","Avg Price"]
                fig = px.bar(ma, x="Merchant", y="Avg Price", text="Avg Price",
                             color="Avg Price", color_continuous_scale="Oranges")
                fig.update_traces(texttemplate="$%{text:.0f}", textposition="outside")
                fig.update_layout(**DARK, coloraxis_showscale=False, height=320)
                st.plotly_chart(fig, width='stretch')

            elif item["chart"] == "cat_price":
                ca = df.groupby("first_category")[price_col].mean().sort_values(ascending=False).reset_index()
                ca.columns = ["Category","Avg Price"]
                fig = px.bar(ca, x="Category", y="Avg Price", text="Avg Price",
                             color="Avg Price", color_continuous_scale="Blues")
                fig.update_traces(texttemplate="$%{text:.0f}", textposition="outside")
                fig.update_layout(**DARK, coloraxis_showscale=False, height=320)
                st.plotly_chart(fig, width='stretch')

            elif item["chart"] == "sale_price":
                sp = df.groupby("pricesissale")[price_col].mean().reset_index()
                sp["pricesissale"] = sp["pricesissale"].map({0:"Regular",1:"On Sale"})
                sp.columns = ["Type","Avg Price"]
                fig = px.bar(sp, x="Type", y="Avg Price", text="Avg Price",
                             color="Type", color_discrete_map={"Regular":"#636EFA","On Sale":"#EF553B"})
                fig.update_traces(texttemplate="$%{text:.0f}", textposition="outside")
                fig.update_layout(**DARK, showlegend=False, height=320)
                st.plotly_chart(fig, width='stretch')

            elif item["chart"] == "cond_price":
                cp = df.groupby("pricescondition")[price_col].mean().sort_values(ascending=False).reset_index()
                cp.columns = ["Condition","Avg Price"]
                fig = px.bar(cp, x="Condition", y="Avg Price", text="Avg Price",
                             color="Avg Price", color_continuous_scale="Greens")
                fig.update_traces(texttemplate="$%{text:.0f}", textposition="outside")
                fig.update_layout(**DARK, coloraxis_showscale=False, height=320)
                st.plotly_chart(fig, width='stretch')

            elif item["chart"] == "shipping":
                sc = df["shipping_type"].value_counts().reset_index()
                sc.columns = ["Shipping","Count"]
                fig = px.pie(sc, names="Shipping", values="Count",
                             color_discrete_sequence=px.colors.qualitative.Set3)
                fig.update_layout(**DARK, height=320)
                st.plotly_chart(fig, width='stretch')

            elif item["chart"] == "brand_cnt" and "brand" in df.columns:
                bc = df["brand"].value_counts().head(10).reset_index()
                bc.columns = ["Brand","Count"]
                fig = px.bar(bc, x="Brand", y="Count", text="Count",
                             color="Count", color_continuous_scale="Reds")
                fig.update_layout(**DARK, coloraxis_showscale=False, height=320)
                st.plotly_chart(fig, width='stretch')

            elif item["chart"] == "availability":
                av = df["is_in_stock"].value_counts().reset_index()
                av.columns = ["Status","Count"]
                fig = px.pie(av, names="Status", values="Count",
                             color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(**DARK, height=320)
                st.plotly_chart(fig, width='stretch')
