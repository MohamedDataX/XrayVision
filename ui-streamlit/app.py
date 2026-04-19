"""
=============================================================================
XRAYVISION - Streamlit UI  (app.py)
=============================================================================
Interface web sobre et professionnelle — détection d'objets X-ray
Imports le style depuis style_css.py
"""

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import requests
import io

from style_css import CSS_STYLE

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="XrayVision",
    page_icon=None,
    layout="centered"
)

# Injection du CSS global
st.markdown(CSS_STYLE, unsafe_allow_html=True)

API_URL = "http://localhost:8000"

CLASS_COLORS = {
    "Gun":              "#D94F4F",
    "Knife":            "#FF3232",
    "Bullet":           "#FF6400",
    "Razor_blade":      "#FF9600",
    "Scissors":         "#00C864",
    "Lighter":          "#00C8FF",
    "Pressure_vessel":  "#E8E832",
    "Wrench":           "#969696",
    "Pliers":           "#6464FF",
    "Hammer":           "#C86432",
    "Screwdriver":      "#32C832",
    "Battery":          "#FFC800",
    "Bat":              "#9632C8",
    "Saw_blade":        "#C83264",
    "Fireworks":        "#FF64FF",
    "Dart":             "#64FFC8",
    "Shuriken":         "#C8C800",
}

DANGEROUS_CLASSES = {"Gun", "Knife", "Bullet", "Razor_blade"}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def call_predict_api(image_bytes: bytes, filename: str) -> dict:
    try:
        files = {"file": (filename, image_bytes, "image/png")}
        r = requests.post(f"{API_URL}/predict", files=files, timeout=30)
        return r.json() if r.status_code == 200 else {"error": f"API {r.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Impossible de se connecter à l'API."}
    except Exception as e:
        return {"error": str(e)}


def draw_detections(image: Image.Image, detections: list) -> Image.Image:
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        class_name = det["class"]
        confidence = det["confidence"]
        bbox = det.get("bbox_pixels", det.get("bbox", []))
        if len(bbox) < 4:
            continue
        color = CLASS_COLORS.get(class_name, "#FFFFFF")
        x1, y1, x2, y2 = bbox[:4]
        outline_width = 3 if det.get("is_dangerous") else 2
        draw.rectangle([x1, y1, x2, y2], outline=color, width=outline_width)
        label = f"{class_name} {confidence:.0%}"
        tb = draw.textbbox((x1, y1), label, font=font)
        th, tw = tb[3] - tb[1], tb[2] - tb[0]
        draw.rectangle([x1, y1 - th - 4, x1 + tw + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - th - 2), label, fill="white", font=font)
    return img_draw


def render_detection_card(det: dict) -> str:
    class_name  = det["class"]
    confidence  = det["confidence"]
    is_dangerous = det.get("is_dangerous", False)
    color       = CLASS_COLORS.get(class_name, "#7a8d9e")
    card_class  = "detection-card dangerous" if is_dangerous else "detection-card"
    label_class = "detection-label dangerous" if is_dangerous else "detection-label"
    return f"""
    <div class="{card_class}">
        <div style="display:flex;align-items:center;gap:0.6rem;">
            <div class="detection-dot" style="background:{color};"></div>
            <span class="{label_class}">{class_name}</span>
        </div>
        <span class="detection-confidence">{confidence:.1%}</span>
    </div>
    """


def render_legend_item(class_name: str, color: str) -> str:
    marker = "◆" if class_name in DANGEROUS_CLASSES else "·"
    return f"""
    <div class="legend-item">
        <div class="legend-dot" style="background:{color};"></div>
        <span>{marker} {class_name}</span>
    </div>
    """


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### Status")
    api_ok = check_api_health()
    badge_class = "status-badge online" if api_ok else "status-badge offline"
    badge_label = "API Online" if api_ok else "API Offline"
    st.markdown(
        f'<div class="{badge_class}"><div class="status-indicator"></div>{badge_label}</div>',
        unsafe_allow_html=True
    )

    if not api_ok:
        st.markdown(
            "<p style='font-size:0.75rem;color:#5a6a7a;'>Lancez : "
            "<code>python main.py</code></p>",
            unsafe_allow_html=True
        )

    st.markdown("### Classes détectables")
    legend_html = "".join(
        render_legend_item(cls, color) for cls, color in CLASS_COLORS.items()
    )
    st.markdown(legend_html, unsafe_allow_html=True)


# ============================================================================
# MAIN CONTENT
# ============================================================================
st.title("XrayVision")
st.markdown(
    "<p style='color:#5a6a7a;font-size:0.85rem;margin-top:-0.5rem;'>"
    "Détection d'objets dans les images X-ray — SSD-CNN-256</p>",
    unsafe_allow_html=True
)
st.markdown("---")

if not api_ok:
    st.markdown(
        "<div style='border:1px solid #252a30;border-left:3px solid #d94f4f;"
        "background:rgba(217,79,79,0.06);border-radius:4px;padding:1rem;'>"
        "<span style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;"
        "color:#e87a7a;'>SERVICE UNAVAILABLE</span><br>"
        "<span style='font-size:0.8rem;color:#7a8d9e;'>L'API n'est pas démarrée.</span>"
        "</div>",
        unsafe_allow_html=True
    )
    st.code("cd inference-api/src && python main.py", language="bash")
    st.stop()

# ── Upload ─────────────────────────────────────────────────────────────────
st.markdown("### Upload")
uploaded_file = st.file_uploader(
    "Image X-ray (PNG / JPG)",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original**", unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    if st.button("Analyser", use_container_width=True):
        with st.spinner("Analyse en cours…"):
            result = call_predict_api(uploaded_file.getvalue(), uploaded_file.name)

        if "error" in result:
            st.markdown(
                f"<div style='border-left:3px solid #d94f4f;background:rgba(217,79,79,0.07);"
                f"padding:0.8rem 1rem;border-radius:4px;font-size:0.82rem;color:#e87a7a;'>"
                f"Erreur : {result['error']}</div>",
                unsafe_allow_html=True
            )
        else:
            detections = result.get("detections", [])

            with col2:
                st.markdown("**Détections**", unsafe_allow_html=True)
                if detections:
                    st.image(draw_detections(image, detections), use_container_width=True)
                else:
                    st.image(image, use_container_width=True)

            st.markdown("---")

            # ── Résultat global ───────────────────────────────────────────
            if result.get("has_dangerous"):
                items = ", ".join(result["dangerous_items"])
                st.markdown(
                    f"<div style='border-left:3px solid #d94f4f;background:rgba(217,79,79,0.08);"
                    f"padding:0.9rem 1.2rem;border-radius:4px;margin-bottom:1rem;'>"
                    f"<span style='font-family:IBM Plex Mono,monospace;font-size:0.78rem;"
                    f"letter-spacing:0.1em;text-transform:uppercase;color:#d94f4f;'>"
                    f"⚠ ALERTE — Objets dangereux</span><br>"
                    f"<span style='font-size:0.85rem;color:#f0a0a0;'>{items}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div style='border-left:3px solid #3ab07a;background:rgba(58,176,122,0.07);"
                    "padding:0.9rem 1.2rem;border-radius:4px;margin-bottom:1rem;'>"
                    "<span style='font-family:IBM Plex Mono,monospace;font-size:0.78rem;"
                    "letter-spacing:0.1em;text-transform:uppercase;color:#3ab07a;'>"
                    "Aucun objet dangereux détecté</span>"
                    "</div>",
                    unsafe_allow_html=True
                )

            # ── Cartes de détection ───────────────────────────────────────
            if detections:
                st.markdown(
                    f"<p style='font-family:IBM Plex Mono,monospace;font-size:0.72rem;"
                    f"letter-spacing:0.1em;text-transform:uppercase;color:#5a6a7a;"
                    f"margin-bottom:0.6rem;'>{len(detections)} détection(s)</p>",
                    unsafe_allow_html=True
                )
                cards_html = "".join(render_detection_card(d) for d in detections)
                st.markdown(cards_html, unsafe_allow_html=True)
            else:
                st.markdown(
                    "<p style='color:#5a6a7a;font-size:0.85rem;'>Aucun objet détecté.</p>",
                    unsafe_allow_html=True
                )

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='xray-footer'>XrayVision · SSD-CNN-256 · Détection X-ray</div>",
    unsafe_allow_html=True
)