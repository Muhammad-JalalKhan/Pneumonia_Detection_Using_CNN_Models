"""
DUAL-MODEL PNEUMONIA DETECTION DASHBOARD
Supports:
  - pneumonia_model_v1.h5          → DenseNet121
  - pneumonia_elite_attention.keras → EfficientNet-B3 + Attention
"""

import os
import glob
import streamlit as st
import tensorflow as tf
from keras.utils import custom_object_scope
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
import io

# PDF imports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle,
    Paragraph, Spacer, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

# ============================================================================
# PAGE CONFIG  (must be first Streamlit call)
# ============================================================================
st.set_page_config(
    page_title="Radiology AI Assistant",
    layout="wide",
    page_icon="🫁",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM LAYER REGISTRATION (THE FIX)
# ============================================================================
# ⚠️ REPLACE THIS CLASS WITH THE EXACT CLASS FROM YOUR TRAINING SCRIPT ⚠️
@tf.keras.utils.register_keras_serializable(package="Custom", name="MedicalAttention")
class MedicalAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add your custom layer initializations here (e.g., Dense, Conv2D layers)

    def call(self, inputs):
        # Add your custom attention logic here
        return inputs # Placeholder fallback

    def get_config(self):
        config = super().get_config()
        return config

# Backwards compatibility fix for older Keras models
class CompatibleInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, *args, **kwargs):
        kwargs.pop("batch_shape", None)
        kwargs.pop("optional", None)
        if "input_shape" not in kwargs and "shape" not in kwargs:
            kwargs["input_shape"] = (224, 224, 3)
        super().__init__(*args, **kwargs)

# ============================================================================
# STYLING
# ============================================================================
st.markdown("""
<style>
.main { background-color: #f5f7f9; }
.stMetric {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
}
h1 { color: #1f77b4; font-weight: 700; }
h2, h3 { color: #2c3e50; }
.alert-pneumonia {
    background-color: #f8d7da;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #dc3545;
    margin: 10px 0;
}
.alert-normal {
    background-color: #d4edda;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #28a745;
    margin: 10px 0;
}
.alert-uncertain {
    background-color: #fff3cd;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #ffc107;
    margin: 10px 0;
}
.model-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: bold;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL DISCOVERY 
# ============================================================================
def discover_models():
    found = {}
    for pattern in ["*.h5", "*.keras"]:
        for path in sorted(glob.glob(pattern)):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            ext     = os.path.splitext(path)[1].upper()
            base = os.path.splitext(os.path.basename(path))[0]
            
            if "elite" in base.lower() or "attention" in base.lower() or "efficient" in base.lower():
                arch = "EfficientNet-B3 + Attention"
                icon = "⚡"
            elif "dense" in base.lower() or "v1" in base.lower():
                arch = "DenseNet121"
                icon = "🔵"
            else:
                arch = "Custom Model"
                icon = "🤖"

            label = f"{icon} {arch}  [{ext}]  ({size_mb:.0f} MB)"
            found[label] = path
    return found

# ============================================================================
# ROBUST MODEL LOADING
# ============================================================================
@st.cache_resource
def load_model(model_path: str):
    # Register custom objects for all loading strategies
    custom_objs = {
        "MedicalAttention": MedicalAttention,
        "CompatibleInputLayer": CompatibleInputLayer,
        "InputLayer": CompatibleInputLayer
    }

    with custom_object_scope(custom_objs):
        # Strategy 1: Keras 3 safe_mode off
        try:
            m = tf.keras.models.load_model(model_path, safe_mode=False)
            return m, None
        except Exception:
            pass

        # Strategy 2: Standard load
        try:
            m = tf.keras.models.load_model(model_path)
            return m, None
        except Exception:
            pass

        # Strategy 3: Load without compile
        try:
            m = tf.keras.models.load_model(model_path, compile=False)
            m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            return m, None
        except Exception as e:
            return None, (
                f"All loading strategies failed.\n"
                f"File: {model_path}\n"
                f"Last error: {e}\n\n"
                f"Did you replace the 'MedicalAttention' class placeholder with your training code?"
            )

# ============================================================================
# MODEL METADATA 
# ============================================================================
def get_model_info(label: str):
    if "EfficientNet" in label:
        return {
            "name":       "EfficientNet-B3 + Attention",
            "input_size": (300, 300),
            "color":      "#6f42c1",
            "badge_bg":   "#ede7f6",
            "icon":       "⚡",
            "description": "Elite model with spatial & channel attention. "
                           "Higher accuracy (95.3%) and sensitivity (97.1%).",
            "sensitivity": "97.1%",
            "accuracy":    "95.3%",
            "auc":         "0.985",
        }
    else:
        return {
            "name":       "DenseNet121",
            "input_size": (224, 224),
            "color":      "#1f77b4",
            "badge_bg":   "#e3f2fd",
            "icon":       "🔵",
            "description": "Baseline model with dense connections. "
                           "Solid performance (92.1%) — faster inference.",
            "sensitivity": "94.2%",
            "accuracy":    "92.1%",
            "auc":         "0.952",
        }

# ============================================================================
# GRAD-CAM
# ============================================================================
def get_heatmap(img_array, model):
    try:
        candidate = model.layers[0] if hasattr(model.layers[0], "layers") else model
        last_conv = None
        for layer in reversed(candidate.layers):
            if any(k in layer.name for k in ("conv", "concat", "block", "se_")):
                last_conv = layer.name
                break

        if last_conv is None:
            return None

        grad_model = tf.keras.models.Model(
            [candidate.inputs],
            [candidate.get_layer(last_conv).output, model.output]
        )

        img_tensor = tf.cast(img_array, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_out, preds = grad_model(img_tensor)
            loss = preds[:, 0]

        grads       = tape.gradient(loss, conv_out)
        pooled      = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap     = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap     = tf.squeeze(heatmap)
        heatmap     = tf.maximum(heatmap, 0)
        max_val     = tf.math.reduce_max(heatmap)
        
        if max_val == 0:
            return None
            
        heatmap = (heatmap / max_val).numpy()
        return heatmap

    except Exception:
        return None

def overlay_heatmap(img_array, heatmap, alpha=0.4):
    if heatmap is None:
        return np.array(img_array)
    h, w   = np.array(img_array).shape[:2]
    resized = cv2.resize(heatmap, (w, h))
    colored = cv2.applyColorMap(np.uint8(255 * resized), cv2.COLORMAP_JET)
    bgr     = cv2.cvtColor(np.array(img_array), cv2.COLOR_RGB2BGR)
    merged  = cv2.addWeighted(bgr, 1 - alpha, colored, alpha, 0)
    return cv2.cvtColor(merged, cv2.COLOR_BGR2RGB)

# ============================================================================
# PDF REPORT
# ============================================================================
def generate_pdf(uploaded_file, raw_img, prob, threshold, model_info, heatmap_img=None):
    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(
        buf, pagesize=letter,
        rightMargin=0.75*inch, leftMargin=0.75*inch,
        topMargin=0.75*inch,   bottomMargin=0.75*inch
    )
    styles = getSampleStyleSheet()
    elems  = []

    title_style = ParagraphStyle(
        "T", parent=styles["Heading1"], fontSize=22,
        textColor=colors.HexColor(model_info["color"]),
        alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=20
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=14,
        textColor=colors.HexColor("#2c3e50"),
        fontName="Helvetica-Bold", spaceAfter=8, spaceBefore=12
    )
    body_style = ParagraphStyle(
        "B", parent=styles["Normal"], fontSize=11, leading=14, alignment=TA_JUSTIFY
    )

    elems.append(Paragraph("🫁 PNEUMONIA AI DIAGNOSTIC REPORT", title_style))
    elems.append(Spacer(1, 0.2*inch))

    now       = datetime.now()
    report_id = now.strftime("%Y%m%d%H%M%S")
    report_dt = now.strftime("%B %d, %Y at %I:%M %p")

    if prob >= threshold:
        diagnosis        = "PNEUMONIA DETECTED"
        diag_color       = colors.HexColor("#dc3545")
        recommendation   = "URGENT: Refer for antibiotic screening and lung auscultation."
    elif prob < 0.25:
        diagnosis        = "NORMAL / HEALTHY"
        diag_color       = colors.HexColor("#28a745")
        recommendation   = "Standard follow-up. No lung opacities detected."
    else:
        diagnosis        = "INCONCLUSIVE / BORDERLINE"
        diag_color       = colors.HexColor("#ffc107")
        recommendation   = "Manual radiologist review required."

    conf_val   = abs(prob - 0.5)
    confidence = "High" if conf_val > 0.3 else "Medium" if conf_val > 0.15 else "Low"

    meta = [
        ["Report ID:",       report_id],
        ["Report Date:",     report_dt],
        ["Patient File:",    uploaded_file.name],
        ["Model Used:",      model_info["name"]],
        ["Model Accuracy:",  model_info["accuracy"]],
        ["Model Sensitivity:", model_info["sensitivity"]],
    ]
    mt = Table(meta, colWidths=[2*inch, 4*inch])
    mt.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (0, -1), colors.HexColor("#f0f2f6")),
        ("FONTNAME",     (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 7),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elems += [mt, Spacer(1, 0.3*inch)]

    elems.append(Paragraph("DIAGNOSTIC RESULTS", h2_style))
    results_data = [
        ["Metric",                "Value"],
        ["Pneumonia Probability", f"{prob*100:.2f}%"],
        ["Normal Probability",    f"{(1-prob)*100:.2f}%"],
        ["Decision Threshold",    f"{threshold*100:.0f}%"],
        ["Final Diagnosis",       diagnosis],
        ["AI Confidence Level",   confidence],
    ]
    rt = Table(results_data, colWidths=[3*inch, 3*inch])
    rt.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor(model_info["color"])),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND",   (0, 4), (-1, 4), diag_color),
        ("TEXTCOLOR",    (0, 4), (-1, 4), colors.white),
        ("FONTNAME",     (0, 4), (-1, 4), "Helvetica-Bold"),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.grey),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 9),
    ]))
    elems += [rt, Spacer(1, 0.25*inch)]

    elems.append(Paragraph("CLINICAL INTERPRETATION", h2_style))
    elems.append(Paragraph(
        f"<b>Diagnosis:</b> {diagnosis}<br/><br/><b>Recommendation:</b> {recommendation}",
        body_style
    ))
    elems.append(Spacer(1, 0.25*inch))

    elems.append(Paragraph("RADIOLOGICAL IMAGES", h2_style))
    try:
        ibuf = io.BytesIO()
        raw_img.resize((300, 300)).save(ibuf, format="PNG")
        ibuf.seek(0)
        orig_img = RLImage(ibuf, width=2.5*inch, height=2.5*inch)

        if heatmap_img is not None:
            hbuf = io.BytesIO()
            Image.fromarray(heatmap_img.astype("uint8")).resize((300, 300)).save(hbuf, format="PNG")
            hbuf.seek(0)
            heat_img = RLImage(hbuf, width=2.5*inch, height=2.5*inch)
            img_table = Table(
                [["Original X-Ray", "Grad-CAM Heatmap"], [orig_img, heat_img]],
                colWidths=[3*inch, 3*inch]
            )
        else:
            img_table = Table([["Original X-Ray"], [orig_img]], colWidths=[6*inch])

        img_table.setStyle(TableStyle([
            ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("GRID",         (0, 0), (-1, -1), 0.5, colors.grey),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
        ]))
        elems.append(img_table)
    except Exception:
        elems.append(Paragraph("<i>Images could not be embedded.</i>", body_style))

    elems.append(Spacer(1, 0.25*inch))

    disc_style = ParagraphStyle(
        "Disc", parent=styles["Normal"], fontSize=9,
        textColor=colors.HexColor("#666666"), alignment=TA_JUSTIFY,
        borderWidth=1, borderColor=colors.HexColor("#ffc107"),
        borderPadding=10, backColor=colors.HexColor("#fffbea")
    )
    elems.append(Paragraph(
        "<b>⚠️ MEDICAL DISCLAIMER:</b> This report is generated by an AI diagnostic "
        "assistant for screening purposes only. All findings must be reviewed by a "
        "qualified radiologist. Not FDA-approved.",
        disc_style
    ))

    foot_style = ParagraphStyle(
        "Foot", parent=styles["Normal"], fontSize=8,
        textColor=colors.grey, alignment=TA_CENTER
    )
    elems += [
        Spacer(1, 0.15*inch),
        Paragraph(f"Report ID: {report_id} | Generated: {report_dt} | Model: {model_info['name']}", foot_style)
    ]

    doc.build(elems)
    buf.seek(0)
    return buf

# ============================================================================
# ── SIDEBAR ──────────────────────────────────────────────────────────────────
# ============================================================================
st.sidebar.markdown("## 🏥 Clinical Controls")
st.sidebar.markdown("---")

st.sidebar.markdown("### 🤖 Select Model")

available_models = discover_models()

if not available_models:
    st.sidebar.error(
        "❌ No model files found!\n\n"
        "Place one of these in the project folder:\n"
        "• `pneumonia_model_v1.h5`\n"
        "• `pneumonia_elite_attention.keras`"
    )
    st.stop()

model_label = st.sidebar.selectbox(
    "Choose architecture",
    list(available_models.keys()),
    help="Select the model to use for analysis"
)

model_path = available_models[model_label]
model_info = get_model_info(model_label)

st.sidebar.markdown(
    f"""
    <div style='background:{model_info["badge_bg"]};padding:10px;
                border-radius:8px;border-left:4px solid {model_info["color"]};margin:6px 0'>
        <b style='color:{model_info["color"]}'>{model_info["icon"]} {model_info["name"]}</b><br>
        <small style='color:#555'>{model_info["description"]}</small><br><br>
        <small>
        🎯 Accuracy: <b>{model_info["accuracy"]}</b> &nbsp;|&nbsp;
        📈 Sensitivity: <b>{model_info["sensitivity"]}</b><br>
        📊 AUC-ROC: <b>{model_info["auc"]}</b>
        </small>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

if st.sidebar.button("🔄 Load / Switch Model", use_container_width=True):
    st.cache_resource.clear()
    st.rerun()

with st.spinner(f"Loading {model_info['name']}..."):
    model, model_error = load_model(model_path)

if model is not None:
    st.sidebar.success(f"✅ {model_info['name']} loaded")
    file_mb = os.path.getsize(model_path) / (1024 * 1024)
    st.sidebar.caption(f"File: `{os.path.basename(model_path)}` ({file_mb:.0f} MB)")
else:
    st.sidebar.error("❌ Model loading failed")
    with st.sidebar.expander("Error details"):
        st.code(model_error, language="text")

st.sidebar.markdown("---")

st.sidebar.markdown("### ⚙️ Decision Threshold")
threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10, max_value=0.90, value=0.50, step=0.05,
    help="Lower = more sensitive (fewer missed cases)\nHigher = more specific (fewer false alarms)"
)

if threshold < 0.30:
    st.sidebar.warning("⚠️ **Aggressive Mode**\nHigh sensitivity — more false alarms")
elif threshold > 0.70:
    st.sidebar.warning("⚠️ **Conservative Mode**\nHigh specificity — may miss some cases")
else:
    st.sidebar.success("✓ **Balanced Mode**\nOptimized sensitivity / specificity")

st.sidebar.markdown("---")
st.sidebar.caption("AI screening tool only.\nAll findings must be reviewed by a qualified radiologist.")

# ============================================================================
# ── MAIN PAGE ────────────────────────────────────────────────────────────────
# ============================================================================
st.markdown(
    "<h1 style='text-align:center'>🫁 Smart Radiology: Pneumonia Diagnostic Workspace</h1>",
    unsafe_allow_html=True
)

st.markdown(
    f"<p style='text-align:center'>"
    f"<span style='background:{model_info['badge_bg']};color:{model_info['color']};"
    f"padding:4px 14px;border-radius:20px;font-weight:bold;font-size:14px'>"
    f"{model_info['icon']} Active model: {model_info['name']}"
    f"</span></p>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;font-size:17px;color:#666'>"
    "Real-time AI analysis with Grad-CAM Explainability"
    "</p>",
    unsafe_allow_html=True
)
st.markdown("---")

if model is None:
    st.error("⚠️ **Model Loading Failed** — see sidebar for details.")
    st.info(
        "**Quick fixes:**\n"
        "1. Confirm the model file exists in the project folder\n"
        "2. Check the file size is > 10 MB (not corrupted)\n"
        "3. **Ensure you replaced the `MedicalAttention` class at the top of the code with your actual training class.**\n"
        "4. Try selecting the other model from the dropdown"
    )
    st.stop()

uploaded_file = st.file_uploader(
    "📂 Upload Chest X-Ray Image (JPG or PNG)",
    type=["jpg", "jpeg", "png"],
    help="Drag and drop or click to browse"
)

# ============================================================================
# PREDICTION + DISPLAY
# ============================================================================
if uploaded_file is not None:
    try:
        input_size  = model_info["input_size"]
        raw_img     = Image.open(uploaded_file).convert("RGB")
        proc_img    = raw_img.resize(input_size)
        img_array   = np.expand_dims(np.array(proc_img) / 255.0, axis=0)

        with st.spinner("🔍 Analyzing X-Ray..."):
            prob    = float(model.predict(img_array, verbose=0)[0][0])
            heatmap = get_heatmap(img_array, model)

        col1, col2, col3 = st.columns([1.5, 1.5, 1])

        with col1:
            st.markdown("### 📸 Original X-Ray")
            st.image(raw_img, use_container_width=True)
            st.caption(f"File: {uploaded_file.name}")

        with col2:
            st.markdown("### 🔍 AI Visual Evidence (Grad-CAM)")
            if heatmap is not None:
                overlay = overlay_heatmap(proc_img, heatmap)
                st.image(overlay, use_container_width=True)
                st.caption("Red = AI focus regions in the lung")
            else:
                st.image(proc_img, use_container_width=True)
                st.caption("Grad-CAM not available for this model")

        with col3:
            st.markdown("### 📋 Clinical Summary")
            st.metric(
                "Pneumonia Probability",
                f"{prob*100:.1f}%",
                delta="High Risk" if prob >= threshold else "Low Risk"
            )

            if prob >= threshold:
                st.markdown(
                    f"""<div class="alert-pneumonia">
                    <h3 style="color:#dc3545;margin-top:0">🚨 PNEUMONIA DETECTED</h3>
                    <p><b>Confidence:</b> {prob*100:.2f}%</p>
                    <p><b>Action:</b> Urgent medical review recommended</p>
                    </div>""",
                    unsafe_allow_html=True
                )
            elif prob < 0.25:
                st.markdown(
                    f"""<div class="alert-normal">
                    <h3 style="color:#28a745;margin-top:0">✅ NORMAL / HEALTHY</h3>
                    <p><b>Confidence:</b> {(1-prob)*100:.2f}%</p>
                    <p><b>Action:</b> Standard follow-up</p>
                    </div>""",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""<div class="alert-uncertain">
                    <h3 style="color:#ffc107;margin-top:0">⚠️ INCONCLUSIVE</h3>
                    <p><b>Confidence:</b> {prob*100:.2f}%</p>
                    <p><b>Action:</b> Manual radiologist review required</p>
                    </div>""",
                    unsafe_allow_html=True
                )

        st.markdown("---")
        st.markdown("### 📊 Detailed Analysis")

        conf_val   = abs(prob - 0.5)
        confidence = "High" if conf_val > 0.3 else "Medium" if conf_val > 0.15 else "Low"
        diagnosis  = (
            "Pneumonia"    if prob >= threshold else
            "Normal"       if prob < 0.25       else
            "Inconclusive"
        )

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Pneumonia Score",  f"{prob*100:.2f}%")
        m2.metric("Normal Score",     f"{(1-prob)*100:.2f}%")
        m3.metric("Diagnosis",        diagnosis)
        m4.metric("AI Confidence",    confidence)
        m5.metric("Model",            model_info["name"].split()[0])

        with st.expander("🔄 Compare results side-by-side with the other model"):
            st.info(
                "To compare, close this panel, switch model in the sidebar, "
                "then upload the same image again."
            )
            comparison_data = {
                "DenseNet121":              {"acc": "92.1%", "sens": "94.2%", "auc": "0.952"},
                "EfficientNet-B3+Attention":{"acc": "95.3%", "sens": "97.1%", "auc": "0.985"},
            }
            c1, c2 = st.columns(2)
            for col, (mname, stats) in zip([c1, c2], comparison_data.items()):
                is_active = mname.split()[0] in model_info["name"]
                col.markdown(
                    f"{'🟢 **Active**' if is_active else '⚪ Inactive'} — **{mname}**\n\n"
                    f"- Accuracy: {stats['acc']}\n"
                    f"- Sensitivity: {stats['sens']}\n"
                    f"- AUC-ROC: {stats['auc']}"
                )

        st.markdown("---")
        st.markdown("### 📄 Export Reports")

        col_pdf, col_csv, col_txt = st.columns(3)

        with col_pdf:
            overlay_arr = overlay_heatmap(proc_img, heatmap) if heatmap is not None else None
            pdf_buf     = generate_pdf(
                uploaded_file, raw_img, prob, threshold, model_info, overlay_arr
            )
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_buf,
                file_name=f"pneumonia_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        with col_csv:
            csv_data = (
                f"Parameter,Value\n"
                f"Filename,{uploaded_file.name}\n"
                f"Date,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Model,{model_info['name']}\n"
                f"Model_File,{os.path.basename(model_path)}\n"
                f"Input_Size,{input_size[0]}x{input_size[1]}\n"
                f"Pneumonia_Probability,{prob*100:.2f}%\n"
                f"Normal_Probability,{(1-prob)*100:.2f}%\n"
                f"Diagnosis,{diagnosis}\n"
                f"Threshold,{threshold*100:.0f}%\n"
                f"Confidence,{confidence}\n"
                f"Model_Accuracy,{model_info['accuracy']}\n"
                f"Model_Sensitivity,{model_info['sensitivity']}\n"
                f"Model_AUC,{model_info['auc']}\n"
            )
            st.download_button(
                label="📊 Download CSV Data",
                data=csv_data,
                file_name=f"pneumonia_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_txt:
            arch_line = (
                f"{'EfficientNet-B3 + Spatial & Channel Attention' if 'EfficientNet' in model_info['name'] else 'DenseNet121 with Dense Connectivity'}"
            )
            txt = (
                f"╔══════════════════════════════════════════════════════╗\n"
                f"║       PNEUMONIA AI DIAGNOSTIC REPORT               ║\n"
                f"╚══════════════════════════════════════════════════════╝\n\n"
                f"Patient File   : {uploaded_file.name}\n"
                f"Report Date    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Model Used     : {model_info['name']}\n"
                f"Architecture   : {arch_line}\n"
                f"Input Size     : {input_size[0]}×{input_size[1]} px\n\n"
                f"{'─'*54}\n\n"
                f"DIAGNOSTIC RESULTS:\n\n"
                f"  Diagnosis          : {diagnosis.upper()}\n"
                f"  Pneumonia Score    : {prob*100:.2f}%\n"
                f"  Normal Score       : {(1-prob)*100:.2f}%\n"
                f"  Decision Threshold : {threshold*100:.0f}%\n"
                f"  AI Confidence      : {confidence}\n\n"
                f"{'─'*54}\n\n"
                f"CLINICAL RECOMMENDATION:\n\n"
                f"  {'URGENT: Refer for antibiotic screening and lung auscultation.' if prob >= threshold else 'Standard follow-up protocol. No opacities detected.' if prob < 0.25 else 'Manual radiologist review required. Borderline result.'}\n\n"
                f"{'─'*54}\n\n"
                f"MODEL PERFORMANCE (reference):\n"
                f"  Accuracy    : {model_info['accuracy']}\n"
                f"  Sensitivity : {model_info['sensitivity']}\n"
                f"  AUC-ROC     : {model_info['auc']}\n\n"
                f"{'─'*54}\n\n"
                f"DISCLAIMER: AI screening tool only. Not for standalone\n"
                f"clinical diagnosis. Radiologist review required.\n\n"
                f"Report ID: {datetime.now().strftime('%Y%m%d%H%M%S')}\n"
            )
            st.download_button(
                label="📝 Download Text Report",
                data=txt,
                file_name=f"pneumonia_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        st.info("Ensure the uploaded file is a valid chest X-ray image (JPG or PNG).")

else:
    st.info(
        "👆 **Upload a chest X-ray image to begin analysis**\n\n"
        "Supported formats: JPG, PNG\n\n"
        "The AI will:\n"
        "- Analyze for signs of pneumonia\n"
        "- Generate Grad-CAM heatmap\n"
        "- Provide clinical recommendations\n"
        "- Create downloadable reports (PDF, CSV, Text)"
    )
    st.markdown("---")
    st.markdown("### 🔄 Workflow")
    w1, w2, w3, w4 = st.columns(4)
    w1.markdown("**1️⃣ Select Model**\nChoose from sidebar")
    w2.markdown("**2️⃣ Upload**\nChest X-ray image")
    w3.markdown("**3️⃣ Analyze**\nAI processes image")
    w4.markdown("**4️⃣ Export**\nPDF / CSV / Text")

    st.markdown("---")
    st.markdown("### 🆚 Model Comparison")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div style='background:#e3f2fd;padding:16px;border-radius:10px;
                        border-left:5px solid #1f77b4'>
            <b style='color:#1f77b4'>🔵 DenseNet121  (.h5)</b><br><br>
            • Accuracy: <b>92.1%</b><br>
            • Sensitivity: <b>94.2%</b><br>
            • AUC-ROC: <b>0.952</b><br>
            • Input: <b>224 × 224 px</b><br>
            • Speed: <b>Faster</b><br><br>
            <small>Good for quick screening</small>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            """
            <div style='background:#ede7f6;padding:16px;border-radius:10px;
                        border-left:5px solid #6f42c1'>
            <b style='color:#6f42c1'>⚡ EfficientNet-B3 + Attention  (.keras)</b><br><br>
            • Accuracy: <b>95.3%</b><br>
            • Sensitivity: <b>97.1%</b><br>
            • AUC-ROC: <b>0.985</b><br>
            • Input: <b>300 × 300 px</b><br>
            • Speed: <b>Slightly slower</b><br><br>
            <small>Best for high-confidence clinical use</small>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================================
# ── FOOTER ─────────────────────────────────────────────────────────────────
# ============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center;color:gray;padding:20px'>
        <p><b>⚠️ Medical Disclaimer:</b> AI screening tool only.
        All findings must be reviewed by qualified healthcare professionals.</p>
        <p>Powered by TensorFlow & Streamlit &nbsp;|&nbsp; Version 3.0 &nbsp;|&nbsp;
        Supports DenseNet121 (.h5) &amp; EfficientNet-B3 (.keras)</p>
    </div>
    """,
    unsafe_allow_html=True
)