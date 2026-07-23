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
# CUSTOM STYLING
# ==================================================
st.markdown("""
    <style>
        /* ---------- Block container: fix padding to a known value so the
           header's negative margin below can cancel it out exactly,
           stretching the header edge-to-edge across the main panel. ---------- */
        .main .block-container {
            padding-left: 2.5rem !important;
            padding-right: 2.5rem !important;
            padding-top: 1.5rem !important;
            max-width: 100% !important;
        }

        /* ---------- Header bar (stretches full width, no icon) ---------- */
        .top-header {
            background: linear-gradient(90deg, #1e3a8a 0%, #2563eb 100%);
            border-radius: 0px;
            padding: 1.6rem 2.5rem;
            margin: -1.5rem -2.5rem 1.8rem -2.5rem;
            width: calc(100% + 5rem);
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0px 4px 14px rgba(37, 99, 235, 0.25);
        }
        .top-header-left {
            display: flex;
            align-items: center;
            gap: 14px;
        }
        .top-header h1 {
            color: white;
            font-size: 1.7rem;
            font-weight: 700;
            margin: 0;
            line-height: 1.2;
        }
        .top-header p {
            color: #dbeafe;
            font-size: 0.9rem;
            margin: 2px 0 0 0;
        }

        /* ---------- Section headings ---------- */
        .subtitle {
            text-align: left;
            color: #6b7280;
            font-size: 0.9rem;
            margin-top: 0px;
            margin-bottom: 1.2rem;
        }
        .footer {
            text-align: center;
            color: #9ca3af;
            font-size: 0.85rem;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #e5e7eb;
        }

        /* ---------- Buttons (force blue, override red primary theme) ---------- */
        div.stButton > button,
        button[kind="primary"] {
            width: 100%;
            font-weight: 600;
            padding: 0.6rem 0;
            border-radius: 8px;
            background-color: #2563eb !important;
            border-color: #2563eb !important;
            color: white !important;
        }
        div.stButton > button:hover,
        button[kind="primary"]:hover {
            background-color: #1d4ed8 !important;
            border-color: #1d4ed8 !important;
        }

        /* ---------- Sliders: force blue theme (override default red) ---------- */
        div[data-baseweb="slider"] > div > div:nth-child(2) {
            background: #2563eb !important;
        }
        div[data-baseweb="slider"] div[role="slider"] {
            background-color: #2563eb !important;
            border-color: #2563eb !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.25) !important;
        }
        div[data-testid="stTickBarMin"], div[data-testid="stTickBarMax"] {
            color: #6b7280 !important;
        }

        /* ---------- Icon + text row: perfect vertical alignment ---------- */
        .icon-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 6px 0;
            min-height: 28px;
        }
        .icon-row img {
            flex-shrink: 0;
            display: block;
        }
        .icon-row .icon-row-label {
            display: flex;
            align-items: center;
            flex: 1;
            line-height: 1.2;
        }
        .icon-row .icon-row-label h1,
        .icon-row .icon-row-label h2,
        .icon-row .icon-row-label h3,
        .icon-row .icon-row-label h4,
        .icon-row .icon-row-label p,
        .icon-row .icon-row-label b,
        .icon-row .icon-row-label span {
            margin: 0;
            padding: 0;
            line-height: 1.2;
        }
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

FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

ICON_SIZE = 24  # single consistent icon size used everywhere


@st.cache_data
def image_to_base64(path: str):
    """Read an image file and return a base64 data URI for inline HTML use.
    Returns None (instead of crashing) if the file can't be found/read."""
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


def icon_text_row(image_path: str, label_html: str, icon_size: int = ICON_SIZE, rounded: bool = True):
    """Render a small icon and text on the exact same line, vertically centered.
    rounded=True crops photos into a circle; rounded=False keeps logos/icons uncropped.
    If the image can't be loaded, shows the text anyway plus a small warning
    so a missing file is obvious instead of silently breaking the layout."""
    img_b64 = image_to_base64(image_path)

    if img_b64 is None:
        st.warning(f"⚠️ Icon not found: `{image_path}` — check the file exists in your assets folder.")
        st.markdown(label_html, unsafe_allow_html=True)
        return

    img_style = (
        f"width:{icon_size}px; height:{icon_size}px; object-fit:cover; border-radius:50%;"
        if rounded else
        f"width:{icon_size}px; height:{icon_size}px; object-fit:contain;"
    )
    # NOTE: fixed alignment — both the icon and the label now live inside a
    # single flexbox (.icon-row) with align-items:center, so icons of the
    # same ICON_SIZE always line up with text regardless of heading tag
    # (h3/h4/b/span all get margin/line-height reset via CSS above).
    st.markdown(
        f"""
        <div class="icon-row">
            <img src="{img_b64}" style="{img_style}">
            <div class="icon-row-label">
                {label_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==================================================
# TOP HEADER BAR (full-width, no icon)
# ==================================================
st.markdown(
    """
    <div class="top-header">
        <div class="top-header-left">
            <div>
                <h1>IRIS FLOWER SPECIES CLASSIFIER</h1>
                <p>Predict the species of an iris flower using a K-Nearest Neighbors model</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ==================================================
# SIDEBAR — INFO PANEL + DATASET (moved here from main area)
# ==================================================
with st.sidebar:
    icon_text_row("assets/info.svg", "<h3>About</h3>", icon_size=24, rounded=False)

    st.write(
        "This app classifies iris flowers into one of three species "
        "based on four physical measurements, using a trained "
        "**K-Nearest Neighbors (KNN)** model."
    )

    st.divider()
    st.subheader("Species")

    for name in FLOWER_NAMES:
        icon_text_row(FLOWER_IMAGES[name], f"<b>{name}</b>", icon_size=26)

    
    st.divider()

    # ---- Dataset section, moved from main body into the sidebar ----
    icon_text_row(
        "assets/database.svg",
        "<h4>Dataset</h4>",
        icon_size=22,
        rounded=False
    )

    st.info("""
**150 Flower Samples**

**3 Species**

**4 Features**
""")
    st.divider()
    st.caption("Model: KNeighborsClassifier • Scaler: StandardScaler")

# ==================================================
# MAIN LAYOUT — Measurements form (left) beside Prediction (right)
# ==================================================
main_left, main_right = st.columns([1, 1], gap="large")

# --------------------------------------------------
# LEFT COLUMN: Enter Flower Measurements
# --------------------------------------------------
with main_left:
    st.subheader("Enter Flower Measurements")
    st.markdown('<p class="subtitle">Use the sliders to set each value.</p>', unsafe_allow_html=True)

    sepal_length = st.slider(
        "Sepal Length (cm)", min_value=4.3, max_value=7.9, value=5.1, step=0.1
    )
    sepal_width = st.slider(
        "Sepal Width (cm)", min_value=2.0, max_value=4.4, value=3.5, step=0.1
    )
    petal_length = st.slider(
        "Petal Length (cm)", min_value=1.0, max_value=6.9, value=1.4, step=0.1
    )
    petal_width = st.slider(
        "Petal Width (cm)", min_value=0.1, max_value=2.5, value=0.2, step=0.1
    )

    st.write("")
    predict_clicked = st.button("Predict Species", type="primary")

# --------------------------------------------------
# RIGHT COLUMN: Prediction
# --------------------------------------------------
with main_right:
    st.subheader("Prediction")

    if not predict_clicked:
        st.markdown(
            '<p class="subtitle">Set the measurements on the left and click '
            '<b>Predict Species</b> to see the result here.</p>',
            unsafe_allow_html=True
        )
    else:
        sample = pd.DataFrame(
            [[sepal_length, sepal_width, petal_length, petal_width]],
            columns=FEATURE_COLUMNS
        )

        sample_scaled = scaler.transform(sample)
        prediction = model.predict(sample_scaled)[0]
        predicted_species = FLOWER_NAMES[prediction]

        # Result card — image, heading and caption are all embedded in ONE
        # HTML block so they actually render inside the styled green box
        # (a separate st.image() call would render outside the div).
        result_img_b64 = image_to_base64(FLOWER_IMAGES[predicted_species])
        if not result_img_b64:
            st.warning(f"⚠️ Icon not found: `{FLOWER_IMAGES[predicted_species]}` — check the file exists in your assets folder.")
        result_img_html = (
            f"""<img src="{result_img_b64}" style="
                    width:200px;
                    height:200px;
                    object-fit:cover;
                    border-radius:14px;
                    display:block;
                    margin:0 auto;
                ">"""
            if result_img_b64 else ""
        )
        st.markdown(
            f"""
            <div style="
                background:#f0fdf4;
                border:1px solid #bbf7d0;
                border-radius:18px;
                padding:24px;
                text-align:center;
                box-shadow:0px 4px 12px rgba(0,0,0,0.08);
            ">
                {result_img_html}
                <h2 style="color:#166534; margin:14px 0 4px 0;">
                    {predicted_species}
                </h2>
                <p style="color:#4b5563; margin:0;">
                    Predicted Species
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success("Model Accuracy: 96.67%")

        # Confidence scores, if the model supports it
        if hasattr(model, "predict_proba"):
            st.write("")
            st.subheader("Confidence Breakdown")

            probabilities = model.predict_proba(sample_scaled)[0]
            highest = probabilities.max()

            st.metric("Prediction Confidence", f"{highest:.1%}")

            prob_df = pd.DataFrame({
                "Species": FLOWER_NAMES,
                "Confidence": probabilities
            }).sort_values("Confidence", ascending=False)

            st.bar_chart(prob_df.set_index("Species"))

            # Icon + species name + confidence, on one aligned line using the
            # flexbox helper (avoids the misalignment st.columns caused).
            for _, row in prob_df.iterrows():
                label_html = (
                    f"<div style='display:flex; justify-content:space-between; width:100%;'>"
                    f"<b>{row['Species']}</b>"
                    f"<span style='color:#4b5563;'>{row['Confidence']:.1%}</span>"
                    f"</div>"
                )
                icon_text_row(FLOWER_IMAGES[row["Species"]], label_html, icon_size=26)

        # Entered values recap
        with st.expander("View entered measurements"):
            st.dataframe(sample, use_container_width=True, hide_index=True)

# ==================================================
# FOOTER
# ==================================================
st.markdown(
    """
    <div class="footer">
        Built using Python, Streamlit & Scikit-learn<br>
        Iris Flower Species Classification
    </div>
    """,
    unsafe_allow_html=True
)