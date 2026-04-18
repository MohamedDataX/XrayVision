"""
=============================================================================
XRAYVISION - Streamlit UI
=============================================================================
Interface web simple pour la détection d'objets X-ray
- Upload d'image
- Affichage des bounding boxes
- Résumé des détections
"""

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import requests
import io
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="XrayVision - Détection X-ray",
    page_icon="🔍",
    layout="centered"
)

API_URL = "http://localhost:8000"

# Classes et couleurs
CLASS_COLORS = {
    "Gun": "",           # Rouge
    "Knife": "",         # Rouge clair
    "Bullet": "#FF0000",        # Orange
    "Razor_blade": "#FF0000",   # Orange clair
    "Scissors": "#FF0000",      # Vert
    "Lighter": "#FF0000",       # Cyan
    "Pressure_vessel": "#FF0000", # Jaune
    "Wrench": "#FF0000",        # Gris
    "Pliers": "#FF0000",        # Bleu clair
    "Hammer": "#FF0000",        # Marron
    "Screwdriver": "#FF0000",   # Vert clair
    "Battery": "#FF0000",       # Or
    "Bat": "#FF0000",           # Violet
    "Saw_blade": "#FF0000",     # Rose
    "Fireworks": "#FF0000",     # Magenta
    "Dart": "#FF0000",          # Turquoise
    "Shuriken": "#FF0000",      # Jaune-vert
}

DANGEROUS_CLASSES = {"Gun", "Knife", "Bullet", "Razor_blade"}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def check_api_health() -> bool:
    """Vérifie si l'API est disponible"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def call_predict_api(image_bytes: bytes, filename: str) -> dict:
    """Appelle l'API de prédiction"""
    try:
        files = {"file": (filename, image_bytes, "image/png")}
        response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Impossible de se connecter à l'API. Vérifiez que le service est démarré."}
    except Exception as e:
        return {"error": str(e)}


def draw_detections(image: Image.Image, detections: list) -> Image.Image:
    """Dessine les bounding boxes sur l'image"""
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # Try to use a font, fallback to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        font = ImageFont.load_default()
    
    for det in detections:
        class_name = det["class"]
        confidence = det["confidence"]
        bbox = det.get("bbox_pixels", det.get("bbox", []))
        
        if len(bbox) < 4:
            continue
        
        # Get color
        color = CLASS_COLORS.get(class_name, "#FFFFFF")
        
        # Draw rectangle
        x1, y1, x2, y2 = bbox[:4]
        outline_width = 3 if det.get("is_dangerous", False) else 2
        draw.rectangle([x1, y1, x2, y2], outline=color, width=outline_width)
        
        # Draw label background
        label = f"{class_name} {confidence:.0%}"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_height = text_bbox[3] - text_bbox[1]
        text_width = text_bbox[2] - text_bbox[0]
        draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=color)
        
        # Draw label text
        draw.text((x1 + 2, y1 - text_height - 2), label, fill="white", font=font)
    
    return img_draw


# ============================================================================
# STREAMLIT UI
# ============================================================================
def main():
    # Header
    st.title("🔍 XrayVision")
    st.markdown("**Détection d'objets dangereux dans les images X-ray**")
    st.markdown("---")
    
    # Sidebar - Status
    with st.sidebar:
        st.header("⚙️ Statut")
        
        api_ok = check_api_health()
        if api_ok:
            st.success("✅ API connectée")
        else:
            st.error("❌ API non disponible")
            st.info("Lancez: `cd inference-api/src && python main.py`")
        
        st.markdown("---")
        st.header("📋 Classes (17)")
        
        # Show class legend
        for class_name in CLASS_COLORS:
            danger_icon = "🔴" if class_name in DANGEROUS_CLASSES else "🟢"
            st.markdown(f"{danger_icon} {class_name}")
    
    # Main content
    if not api_ok:
        st.warning("⚠️ L'API n'est pas disponible. Veuillez la démarrer pour utiliser l'application.")
        
        st.code("""
# Démarrer l'API:
cd inference-api/src
pip install -r ../requirements.txt
python main.py
        """, language="bash")
        return
    
    # File upload
    st.header("📤 Upload d'image")
    uploaded_file = st.file_uploader(
        "Choisissez une image X-ray",
        type=["png", "jpg", "jpeg"],
        help="Formats supportés: PNG, JPG, JPEG"
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Image originale")
            st.image(image, use_container_width=True)
        
        # Analyze button
        if st.button("Analyser", type="primary", use_container_width=True):
            with st.spinner("Analyse en cours..."):
                # Call API
                image_bytes = uploaded_file.getvalue()
                result = call_predict_api(image_bytes, uploaded_file.name)
            
            if "error" in result:
                st.error(f"❌ Erreur: {result['error']}")
            else:
                # Show results
                detections = result.get("detections", [])
                
                with col2:
                    st.subheader("Détections")
                    if detections:
                        img_with_boxes = draw_detections(image, detections)
                        st.image(img_with_boxes, use_container_width=True)
                    else:
                        st.image(image, use_container_width=True)
                        st.info("Aucun objet détecté")
                
                # Summary
                st.markdown("---")
                st.header("Résultats")
                
                # Alert if dangerous
                if result.get("has_dangerous", False):
                    st.error(f"🚨 **ALERTE**: Objets dangereux détectés: {', '.join(result['dangerous_items'])}")
                else:
                    st.success("✅ Aucun objet dangereux détecté")
                
                # Detection details
                if detections:
                    st.subheader(f"Détections ({len(detections)})")
                    
                    for i, det in enumerate(detections):
                        class_name = det["class"]
                        confidence = det["confidence"]
                        is_dangerous = det.get("is_dangerous", False)
                        
                        color = CLASS_COLORS.get(class_name, "#FFFFFF")
                        icon = "🔴" if is_dangerous else "🟢"
                        
                        st.markdown(
                            f'{icon} **{class_name}** - Confiance: `{confidence:.1%}` '
                            f'<span style="color:{color}">●</span>',
                            unsafe_allow_html=True
                        )
                else:
                    st.info("Aucun objet détecté dans cette image.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "XrayVision - Détection d'objets X-ray avec SSD-CNN-256"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
