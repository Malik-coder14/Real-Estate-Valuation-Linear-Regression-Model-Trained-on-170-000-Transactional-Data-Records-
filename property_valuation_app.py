"""
Dubai Dream Homes 🏡🌸 — Streamlit property valuation app

Place this file in the same folder as:
  - valuation_model_bundle.joblib
  - cleaned_modelling_dataset.csv

Run:
    pip install streamlit pandas numpy scikit-learn joblib
    streamlit run property_valuation_app.py
"""

from pathlib import Path
import html
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------ Page setup ------------------------------
st.set_page_config(
    page_title="Dubai Dream Homes 🏡",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "valuation_model_bundle.joblib"
DATA_PATH = BASE_DIR / "cleaned_modelling_dataset.csv"

# A soft, light-blue theme with rounded cards and gentle shadows.
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #eaf7ff 0%, #d8efff 48%, #f5fbff 100%);
        color: #18324a;
    }
    [data-testid="stHeader"] { background: rgba(234,247,255,0.88); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #c9eaff 0%, #e8f6ff 100%);
        border-right: 1px solid #b6ddf7;
    }
    [data-testid="stSidebar"] * { color: #173b58; }
    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 24px;
        background: linear-gradient(115deg, rgba(255,255,255,.94), rgba(214,239,255,.94));
        border: 1px solid rgba(126,190,229,.65);
        box-shadow: 0 10px 28px rgba(50,115,160,.12);
        margin-bottom: 1rem;
    }
    .hero h1 { color: #13527b; margin: 0; font-size: clamp(2rem, 4vw, 3.1rem); }
    .hero p { color: #416781; font-size: 1.08rem; margin: .55rem 0 0 0; }
    .mini-card {
        background: rgba(255,255,255,.88);
        border: 1px solid #c4e4f8;
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 5px 18px rgba(46,112,157,.08);
        min-height: 110px;
    }
    .mini-card h3 { margin: 0 0 .35rem 0; color: #155780; font-size: 1.1rem; }
    .mini-card p { margin: 0; color: #4c6b80; }
    .price-card {
        background: linear-gradient(135deg, #ffffff, #dff3ff);
        border: 2px solid #a8dafa;
        border-radius: 24px;
        padding: 1.4rem;
        box-shadow: 0 10px 28px rgba(38,112,164,.14);
        text-align: center;
    }
    .price-label { color: #3c6d8c; font-size: 1.05rem; }
    .price-value { color: #075b8e; font-weight: 800; font-size: clamp(2rem, 4vw, 3rem); line-height: 1.25; }
    .price-range { color: #426e88; font-size: 1.05rem; margin-top: .4rem; }
    .section-title { color: #13527b; font-weight: 800; font-size: 1.45rem; margin: .8rem 0 .3rem 0; }
    .soft-note {
        background: rgba(255,255,255,.7); border-left: 5px solid #69bde9;
        padding: .8rem 1rem; border-radius: 10px; color: #355d77;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.75); padding: 14px 16px;
        border: 1px solid #c5e5f7; border-radius: 16px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1386c5, #42a9df); color: white;
        border: 0; border-radius: 13px; padding: .7rem 1.2rem; font-weight: 700;
        box-shadow: 0 5px 14px rgba(20,126,187,.2);
    }
    .stButton > button:hover { background: linear-gradient(90deg, #0c6fa8, #258fc9); color: white; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Remote images provide visual decoration without requiring extra image files.
HERO_IMAGE = "https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?auto=format&fit=crop&w=1600&q=85"
PROPERTY_IMAGES = [
    "https://images.unsplash.com/photo-1600607687939-ce8a6c25118c?auto=format&fit=crop&w=900&q=85",
    "https://images.unsplash.com/photo-1600607687920-4e2a09cf159d?auto=format&fit=crop&w=900&q=85",
    "https://images.unsplash.com/photo-1600566753086-00f18fb6b3ea?auto=format&fit=crop&w=900&q=85",
]


def money_aed(value: float) -> str:
    return f"AED {value:,.0f}"


def safe_options(series: pd.Series, fallback: list[str]) -> list[str]:
    values = series.dropna().astype(str).str.strip()
    values = sorted(v for v in values.unique().tolist() if v and v.lower() != "nan")
    return values or fallback


@st.cache_resource(show_spinner=False)
def load_model_bundle(path_string: str):
    return joblib.load(path_string)


@st.cache_data(show_spinner=False)
def load_dataset(path_string: str):
    return pd.read_csv(path_string)


# ------------------------------ Load project files ------------------------------
missing_files = [p.name for p in (MODEL_PATH, DATA_PATH) if not p.exists()]
if missing_files:
    st.error(
        "🌷 I couldn't find the required project file(s): " + ", ".join(missing_files)
        + ". Please place this app.py file, valuation_model_bundle.joblib, and "
        "cleaned_modelling_dataset.csv in the same folder."
    )
    st.stop()

try:
    bundle = load_model_bundle(str(MODEL_PATH))
    sales = load_dataset(str(DATA_PATH))
except Exception as exc:
    st.error(f"There was a problem loading the model or dataset: {exc}")
    st.stop()

model = bundle["model"]
features = bundle["features"]
range_pct = float(bundle.get("valuation_range_pct", 16.44))

# ------------------------------ Sidebar ------------------------------
with st.sidebar:
    st.markdown("# 🌸 Hello, Home Hunter! 🏡")
    st.write("Find an estimated value for your Dubai dream property. 😊")
    st.image(PROPERTY_IMAGES[0], use_container_width=True)
    st.markdown("### ✨ Quick guide")
    st.markdown(
        "1. Choose the property's location.\n"
        "2. Enter its type, size and details.\n"
        "3. Click **Estimate My Property**.\n\n"
        "🌼 Tip: use the size in **square feet (sq ft)**."
    )
    st.caption("💙 Estimates are educational guidance, not an official appraisal.")

# ------------------------------ Hero ------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>🏡 Dubai Dream Homes 🌸</h1>
      <p>Your friendly AI-powered property price explorer ✨😊</p>
      <p>Explore an indicative sale value with a little sunshine and a lot of data. ☀️</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Three decorative photo panels.
img_cols = st.columns(3, gap="medium")
photo_captions = ["🌴 Modern villa living", "🌊 Bright, beautiful interiors", "🌷 A place to call home"]
for col, image_url, caption in zip(img_cols, PROPERTY_IMAGES, photo_captions):
    with col:
        st.image(image_url, use_container_width=True)
        st.caption(caption)

st.markdown('<div class="section-title">💎 Tell us about your property</div>', unsafe_allow_html=True)
st.write("Add the details below, then let the valuation model do its work. 🌟")

# ------------------------------ Input controls ------------------------------
areas = safe_options(sales.get("area_name_en", pd.Series(dtype=str)), ["Dubai Marina", "Palm Jumeirah"])
property_types = safe_options(sales.get("property_type_en", pd.Series(dtype=str)), ["Unit", "Villa"])
sub_types = safe_options(sales.get("property_sub_type_en", pd.Series(dtype=str)), ["Flat", "Villa"])
room_options = safe_options(sales.get("rooms_en", pd.Series(dtype=str)), ["Studio", "1 B/R", "2 B/R", "3 B/R", "4 B/R", "5 B/R"])
reg_types = safe_options(sales.get("reg_type_en", pd.Series(dtype=str)), ["Existing Properties", "Off-Plan Properties"])

# Use the most common subtype for the selected property type when possible.
def subtype_options_for(ptype: str) -> list[str]:
    if "property_type_en" in sales.columns and "property_sub_type_en" in sales.columns:
        subset = sales.loc[sales["property_type_en"].astype(str) == ptype, "property_sub_type_en"]
        opts = safe_options(subset, [])
        if opts:
            return opts
    return sub_types

with st.form("valuation_form", clear_on_submit=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        area_name = st.selectbox("📍 Community / area", areas, index=0)
        property_type = st.selectbox("🏘️ Property type", property_types)
        size_sqft = st.number_input("📐 Built-up area (sq ft)", min_value=100.0, max_value=100000.0, value=1200.0, step=50.0)
    with c2:
        subtypes_for_type = subtype_options_for(property_type)
        property_sub_type = st.selectbox("🏠 Property subtype", subtypes_for_type)
        rooms = st.selectbox("🛏️ Bedrooms / rooms", room_options)
        has_parking = st.selectbox("🚗 Parking available?", ["Yes", "No"])
    with c3:
        reg_type = st.selectbox("🗝️ Sale category", reg_types)
        sale_year = st.selectbox("📅 Sale year", list(range(2020, 2026)), index=5)
        sale_month = st.selectbox("🗓️ Sale month", list(range(1, 13)), index=9, format_func=lambda m: pd.Timestamp(2024, m, 1).strftime("%B"))
    submitted = st.form_submit_button("✨ Estimate My Property 🏡", use_container_width=True)

# ------------------------------ Prediction ------------------------------
if submitted:
    input_row = pd.DataFrame([{
        "area_name_en": area_name,
        "property_type_en": property_type,
        "property_sub_type_en": property_sub_type,
        "rooms_en": rooms,
        "reg_type_en": reg_type,
        "has_parking": has_parking,
        "size_sqft": float(size_sqft),
        "sale_year": int(sale_year),
        "sale_month": int(sale_month),
    }], columns=features)

    try:
        # The chosen model was trained on log(actual_worth), so reverse it with exp().
        predicted_log_price = float(model.predict(input_row)[0])
        predicted_price = float(np.exp(predicted_log_price))
        if not np.isfinite(predicted_price) or predicted_price <= 0:
            raise ValueError("The model returned an invalid price. Please check the property details.")

        lower_price = predicted_price * (1 - range_pct / 100)
        upper_price = predicted_price * (1 + range_pct / 100)

        st.markdown("---")
        st.markdown('<div class="section-title">🎉 Your estimated property value</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="price-card">
              <div class="price-label">🌸 Indicative estimated sale price 🌸</div>
              <div class="price-value">{money_aed(predicted_price)}</div>
              <div class="price-range">Illustrative range: <b>{money_aed(lower_price)}</b> – <b>{money_aed(upper_price)}</b></div>
              <p style="color:#5d8198;margin:.8rem 0 0 0;">Made with data, a little AI magic ✨ and lots of hope for your next chapter 😊</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("💰 Estimated value", money_aed(predicted_price))
        m2.metric("⬇️ Lower guide", money_aed(lower_price))
        m3.metric("⬆️ Upper guide", money_aed(upper_price))

        st.markdown('<div class="section-title">🔎 Recent similar sales in the dataset</div>', unsafe_allow_html=True)
        comps = sales.copy()
        # Filter by community and property type, then rank by closest size.
        if "area_name_en" in comps.columns:
            comps = comps[comps["area_name_en"].astype(str) == str(area_name)]
        if "property_type_en" in comps.columns:
            comps = comps[comps["property_type_en"].astype(str) == str(property_type)]
        if "size_sqft" in comps.columns:
            comps = comps.assign(size_difference=(pd.to_numeric(comps["size_sqft"], errors="coerce") - size_sqft).abs())
            comps = comps.sort_values("size_difference")
        display_columns = [c for c in ["area_name_en", "property_type_en", "property_sub_type_en", "rooms_en", "size_sqft", "actual_worth", "sale_year", "sale_month"] if c in comps.columns]
        comps = comps[display_columns].head(5).copy()
        if not comps.empty:
            if "size_sqft" in comps.columns:
                comps["size_sqft"] = pd.to_numeric(comps["size_sqft"], errors="coerce").round(0)
            if "actual_worth" in comps.columns:
                comps["actual_worth"] = pd.to_numeric(comps["actual_worth"], errors="coerce").map(lambda x: money_aed(x) if pd.notna(x) else "—")
            if "sale_month" in comps.columns:
                comps["sale_month"] = pd.to_numeric(comps["sale_month"], errors="coerce").map(lambda x: int(x) if pd.notna(x) else "—")
            st.dataframe(comps, use_container_width=True, hide_index=True)
            st.caption("📌 Comparables are records from the supplied modelling dataset, not live listings or guaranteed direct matches.")
        else:
            st.info("🌼 No similar sales were found for this exact area and property type. The estimate is still based on the trained model, but please treat it with extra care.")

        with st.expander("🧠 How should I interpret this estimate?"):
            st.write(
                "This app uses a trained log-price Linear Regression pipeline and the supplied Dubai transaction dataset. "
                "The range above is a simple illustrative band based on the model's test-set MAPE; it is **not** a formal confidence or prediction interval. "
                "Property condition, floor, view, renovations, exact unit attributes and negotiation circumstances are not fully represented in the data. "
                "For a purchase or sale decision, compare recent transactions and seek a qualified local valuation. 💙"
            )
    except Exception as exc:
        st.error(f"Sorry 😊, the estimate could not be calculated: {exc}")
else:
    st.markdown(
        '<div class="soft-note">🌷 Ready when you are! Fill in the details above and click <b>Estimate My Property</b> to see an indicative value. 😊</div>',
        unsafe_allow_html=True,
    )

# ------------------------------ Footer ------------------------------
st.markdown("---")
footer_cols = st.columns([1, 3, 1])
with footer_cols[1]:
    st.markdown(
        "<p style='text-align:center;color:#47718b;'>🌸 Built for property explorers in Dubai 💙 &nbsp;|&nbsp; 🏡 Dream big, value wisely! ✨</p>",
        unsafe_allow_html=True,
    )
