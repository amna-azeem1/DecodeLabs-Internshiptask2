import streamlit as st
import pandas as pd
import joblib
import base64
from pathlib import Path

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Iris Flower Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# DESIGN TOKENS (single source of truth for the palette)
# ==================================================
PRIMARY = "#2563EB"
PRIMARY_DARK = "#1D4ED8"
BG = "#F8FAFC"
SUCCESS = "#16A34A"
SUCCESS_BG = "#DCFCE7"
TEXT = "#111827"
TEXT_SECONDARY = "#6B7280"
BORDER = "#E5E7EB"

# ==================================================
# CUSTOM STYLING
# ==================================================
st.markdown(f"""
    <style>
        .stApp {{
            background-color: {BG};
        }}

        /* ---------- Hero (compact bar, not a tall block) ---------- */
        .hero {{
            display: flex;
            align-items: baseline;
            gap: 12px;
            padding: 0.9rem 1.2rem;
            background: white;
            border: 1px solid {BORDER};
            border-radius: 12px;
            margin-bottom: 1.5rem;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.03);
        }}
        .hero h1 {{
            font-size: 1.4rem;
            font-weight: 800;
            color: {TEXT};
            margin: 0;
            letter-spacing: -0.3px;
            white-space: nowrap;
        }}
        .hero p {{
            color: {TEXT_SECONDARY};
            font-size: 0.9rem;
            margin: 0;
        }}

        /* ---------- Generic card (for atomic HTML-only blocks) ---------- */
        .card {{
            background: white;
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0px 2px 10px rgba(0,0,0,0.04);
        }}
        .sidebar-card {{
            background: white;
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 16px 18px;
            margin-bottom: 16px;
            box-shadow: 0px 2px 6px rgba(0,0,0,0.03);
        }}
        .sidebar-card-title {{
            font-weight: 700;
            color: {TEXT};
            margin: 0 0 8px 0;
            font-size: 0.95rem;
        }}
        .sidebar-card p {{
            color: {TEXT_SECONDARY};
            font-size: 0.88rem;
            margin: 0;
            line-height: 1.5;
        }}

        /* ---------- Streamlit's real bordered container (used for cards
           that hold live widgets — sliders, buttons, metrics). This is the
           supported way to get an actual bordered box that wraps its
           children, so it never renders as an empty box. ---------- */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            border-radius: 16px !important;
            box-shadow: 0px 2px 10px rgba(0,0,0,0.04);
        }}

        /* ---------- Buttons ---------- */
        div.stButton > button,
        button[kind="primary"] {{
            width: 100%;
            font-weight: 600;
            padding: 0.7rem 0;
            border-radius: 12px !important;
            background: linear-gradient(135deg, {PRIMARY} 0%, {PRIMARY_DARK} 100%) !important;
            border: none !important;
            color: white !important;
            box-shadow: 0px 4px 12px rgba(37, 99, 235, 0.3);
            transition: all 0.15s ease-in-out;
        }}
        div.stButton > button:hover,
        button[kind="primary"]:hover {{
            transform: translateY(-1px);
            box-shadow: 0px 6px 16px rgba(37, 99, 235, 0.4) !important;
        }}

        /* ---------- Sliders: blue theme ---------- */
        div[data-baseweb="slider"] > div > div:nth-child(2) {{
            background: {PRIMARY} !important;
        }}
        div[data-baseweb="slider"] div[role="slider"] {{
            background-color: {PRIMARY} !important;
            border-color: {PRIMARY} !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.25) !important;
        }}

        /* ---------- Custom progress bars ---------- */
        .progress-row {{
            margin-bottom: 16px;
        }}
        .progress-label-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
        }}
        .progress-track {{
            background: #E5E7EB;
            border-radius: 8px;
            height: 10px;
            width: 100%;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            border-radius: 8px;
        }}

        /* ---------- Stat cards ---------- */
        .stat-card {{
            background: white;
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            box-shadow: 0px 2px 10px rgba(0,0,0,0.04);
        }}
        .stat-label {{
            color: {TEXT_SECONDARY};
            font-size: 0.85rem;
            font-weight: 500;
            margin: 0 0 6px 0;
        }}
        .stat-value {{
            color: {TEXT};
            font-size: 1.7rem;
            font-weight: 800;
            margin: 0;
        }}

        /* ---------- Footer ---------- */
        .footer {{
            text-align: center;
            color: {TEXT_SECONDARY};
            font-size: 0.85rem;
            margin-top: 3rem;
            padding-top: 1.2rem;
            border-top: 1px solid {BORDER};
            line-height: 1.6;
        }}
    </style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD MODEL + SCALER
# ==================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("iris_knn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error(
        "⚠️ Model files not found. Make sure `iris_knn_model.pkl` and "
        "`scaler.pkl` are in the same folder as this app, then restart."
    )
    st.stop()

FLOWER_NAMES = ["Setosa", "Versicolor", "Virginica"]

FLOWER_IMAGES = {
    "Setosa": "assets/setosa.jfif",
    "Versicolor": "assets/versicolor.jfif",
    "Virginica": "assets/virginica.jfif"
}

FLOWER_COLORS = {
    "Setosa": "#2563EB",
    "Versicolor": "#7C3AED",
    "Virginica": "#DB2777",
}

FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

# Static reporting values (not recomputed at runtime — reflects your
# training script's held-out test accuracy)
ACCURACY_DISPLAY = "96.67%"
K_VALUE = getattr(model, "n_neighbors", "N/A")


@st.cache_data
def image_to_base64(path: str):
    """Read an image file and return a base64 data URI for inline HTML use.
    Returns None if the file can't be found/read."""
    try:
        ext = Path(path).suffix.lstrip(".").lower()
        if ext == "svg":
            mime = "svg+xml"
        elif ext in ("jpg", "jfif"):
            mime = "jpeg"
        else:
            mime = ext
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        return f"data:image/{mime};base64,{encoded}"
    except Exception:
        return None


def sidebar_card(title: str, content_html: str):
    """One atomic HTML block per card — title + content in a single
    st.markdown call, so it always renders as a real self-contained box
    (never an empty div)."""
    st.markdown(
        f"""
        <div class="sidebar-card">
            <p class="sidebar-card-title">{title}</p>
            <p>{content_html}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def stat_card(label: str, value: str):
    st.markdown(
        f"""
        <div class="stat-card">
            <p class="stat-label">{label}</p>
            <p class="stat-value">{value}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def progress_bar(label: str, pct: float, color: str):
    st.markdown(
        f"""
        <div class="progress-row">
            <div class="progress-label-row">
                <span style="font-weight:600; color:{TEXT};">{label}</span>
                <span style="color:{TEXT_SECONDARY}; font-weight:500;">{pct:.1f}%</span>
            </div>
            <div class="progress-track">
                <div class="progress-fill" style="width:{pct}%; background:{color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==================================================
# HEADER BAR (compact, single-line)
# ==================================================
st.markdown(
    """
    <div class="hero">
        <h1>Iris Flower Classifier</h1>
        <p>AI-powered flower species prediction using K-Nearest Neighbors</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown(
        f"<p style='font-size:1.3rem; font-weight:800; color:{TEXT}; margin-bottom:1.2rem;'> Iris AI</p>",
        unsafe_allow_html=True
    )

    sidebar_card(
        "About",
        "This application predicts the species of an Iris flower using a trained KNN model."
    )

    sidebar_card(
        "Dataset",
        "150 Samples<br>3 Species<br>4 Features"
    )

    sidebar_card(
        "Model",
        f"KNN (K={K_VALUE})<br>Accuracy: {ACCURACY_DISPLAY}"
    )

# ==================================================
# MAIN LAYOUT — Measurements (left) beside Prediction (right)
# ==================================================
main_left, main_right = st.columns([1, 1], gap="large")

# --------------------------------------------------
# LEFT: Flower Measurements (real bordered container — holds live widgets)
# --------------------------------------------------
with main_left:

    st.markdown("""
    <style>
    div[data-testid="stVerticalBlock"]:has(div.measurement-card){
        background:#f3f4f6;
        border:1px solid #e5e7eb;
        border-radius:18px;
        padding:22px;
        box-shadow:0 4px 12px rgba(0,0,0,0.06);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="measurement-card"></div>', unsafe_allow_html=True)

    st.subheader("Flower Measurements")
    st.caption("Adjust the sliders to describe the flower.")

    sepal_length = st.slider("Sepal Length (cm)", 4.3, 7.9, 5.1, 0.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.4, 3.5, 0.1)
    petal_length = st.slider("Petal Length (cm)", 1.0, 6.9, 1.4, 0.1)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2, 0.1)

    st.write("")
    predict_clicked = st.button("Predict Species", type="primary")

# --------------------------------------------------
# RIGHT: Prediction Result (real bordered container)
# --------------------------------------------------
predicted_species = None
probabilities = None
sample = None

with main_right:
    with st.container(border=True):
        st.subheader("Prediction Result")

        if not predict_clicked:
            st.caption("Set the measurements on the left and click **Predict Species** to see results here.")
        else:
            sample = pd.DataFrame(
                [[sepal_length, sepal_width, petal_length, petal_width]],
                columns=FEATURE_COLUMNS
            )
            sample_scaled = scaler.transform(sample)
            prediction = model.predict(sample_scaled)[0]
            predicted_species = FLOWER_NAMES[prediction]

            confidence = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(sample_scaled)[0]
                probabilities = {FLOWER_NAMES[i]: float(p) * 100 for i, p in enumerate(proba)}
                confidence = probabilities[predicted_species]

            result_img_b64 = image_to_base64(FLOWER_IMAGES[predicted_species])
            if not result_img_b64:
                st.warning(f"⚠️ Image not found: `{FLOWER_IMAGES[predicted_species]}` — check your assets folder.")

            img_html = (
                f'<img src="{result_img_b64}" style="width:100%; max-width:280px; height:280px; '
                f'object-fit:cover; border-radius:16px; box-shadow:0 4px 14px rgba(0,0,0,0.12);">'
                if result_img_b64 else ""
            )
            confidence_html = (
                f'<p style="color:{TEXT_SECONDARY}; margin:16px 0 2px 0; font-size:0.9rem;">Confidence</p>'
                f'<p style="font-size:2.1rem; font-weight:800; color:{PRIMARY}; margin:0;">{confidence:.1f}%</p>'
                if confidence is not None else ""
            )

            st.markdown(
                f"""
                <div style="text-align:center;">
                    {img_html}
                    <h2 style="font-size:1.9rem; font-weight:800; color:{TEXT}; margin:16px 0 6px 0;">
                        {predicted_species}
                    </h2>
                    <span style="display:inline-flex; align-items:center; gap:6px;
                                 background:{SUCCESS_BG}; color:{SUCCESS}; font-weight:600;
                                 padding:6px 16px; border-radius:999px; font-size:0.85rem;">
                        Prediction Successful
                    </span>
                    {confidence_html}
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.expander("View entered measurements"):
                st.dataframe(sample, use_container_width=True, hide_index=True)

# ==================================================
# CONFIDENCE BREAKDOWN (full width, custom progress bars)
# ==================================================
st.write("")
st.subheader("Confidence Breakdown")

with st.container(border=True):
    if probabilities is None:
        st.caption("Run a prediction above to see the confidence breakdown for each species.")
    else:
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        for species, pct in sorted_probs:
            progress_bar(species, pct, FLOWER_COLORS[species])

# ==================================================
# STATISTICS ROW
# ==================================================
st.write("")
stat_col1, stat_col2, stat_col3 = st.columns(3)

with stat_col1:
    stat_card("Accuracy", ACCURACY_DISPLAY)
with stat_col2:
    stat_card("K Value", str(K_VALUE))
with stat_col3:
    stat_card("Dataset", "150 Samples")

# ==================================================
# FOOTER
# ==================================================
st.markdown(
    """
    <div class="footer">
        Built using Python • Streamlit • Scikit-learn<br>
        © 2026 Amna Azeem
    </div>
    """,
    unsafe_allow_html=True
)