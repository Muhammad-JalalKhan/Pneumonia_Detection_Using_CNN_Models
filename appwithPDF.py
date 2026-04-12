"""
Complete Pneumonia Detection Dashboard with PDF Report Generation
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
import io

# PDF Generation imports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

# ========================================================================
# PAGE CONFIGURATION
# ========================================================================
st.set_page_config(
    page_title="Radiology AI Assistant", 
    layout="wide",
    page_icon="🫁",
    initial_sidebar_state="expanded"
)

# ========================================================================
# STYLING
# ========================================================================
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
    </style>
    """, unsafe_allow_html=True)

# ========================================================================
# MODEL LOADING
# ========================================================================
@st.cache_resource
def load_elite_model():
    try:
        model = tf.keras.models.load_model('pneumonia_model_v1.h5')
        return model, None
    except Exception as e:
        return None, str(e)

model, model_error = load_elite_model()

# ========================================================================
# GRAD-CAM FUNCTIONS
# ========================================================================
def get_heatmap(img_array, model):
    try:
        base_model = model.layers[0]
        last_conv_layer_name = None
        for layer in reversed(base_model.layers):
            if 'conv' in layer.name or 'concat' in layer.name:
                last_conv_layer_name = layer.name
                break
        
        if last_conv_layer_name is None:
            return None
        
        grad_model = tf.keras.models.Model(
            [base_model.inputs], 
            [base_model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]
        
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    except Exception:
        return None

def overlay_heatmap(img, heatmap, alpha=0.4):
    if heatmap is None:
        return img
    
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    superimposed = cv2.addWeighted(img_bgr, 1-alpha, heatmap_color, alpha, 0)
    result = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
    
    return result

# ========================================================================
# PDF REPORT GENERATION
# ========================================================================
def generate_pdf_report(uploaded_file, raw_img, prob, threshold, heatmap_img=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=0.75*inch, 
                          leftMargin=0.75*inch, topMargin=0.75*inch, bottomMargin=0.75*inch)
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24,
                                textColor=colors.HexColor('#1f77b4'), spaceAfter=30,
                                alignment=TA_CENTER, fontName='Helvetica-Bold')
    
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=16,
                                  textColor=colors.HexColor('#2c3e50'), spaceAfter=12,
                                  spaceBefore=12, fontName='Helvetica-Bold')
    
    normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'], fontSize=11,
                                 leading=14, alignment=TA_JUSTIFY)
    
    # Title
    title = Paragraph("🫁 PNEUMONIA AI DIAGNOSTIC REPORT", title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.3*inch))
    
    # Metadata
    report_date = datetime.now().strftime('%B %d, %Y at %I:%M %p')
    report_id = datetime.now().strftime('%Y%m%d%H%M%S')
    
    metadata = [
        ['Report ID:', report_id],
        ['Report Date:', report_date],
        ['Patient File:', uploaded_file.name],
        ['Analysis Model:', 'DenseNet121 Clinical v2.1']
    ]
    
    metadata_table = Table(metadata, colWidths=[2*inch, 4*inch])
    metadata_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f2f6')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    
    elements.append(metadata_table)
    elements.append(Spacer(1, 0.4*inch))
    
    # Diagnostic Results
    elements.append(Paragraph("DIAGNOSTIC RESULTS", heading_style))
    
    if prob >= threshold:
        diagnosis = "PNEUMONIA DETECTED"
        diagnosis_color = colors.HexColor('#dc3545')
        recommendation = "URGENT: Refer for immediate antibiotic screening and lung auscultation."
    elif prob < 0.25:
        diagnosis = "NORMAL / HEALTHY"
        diagnosis_color = colors.HexColor('#28a745')
        recommendation = "Standard follow-up protocol. No lung opacities detected."
    else:
        diagnosis = "INCONCLUSIVE / BORDERLINE"
        diagnosis_color = colors.HexColor('#ffc107')
        recommendation = "Manual radiologist review required before clinical decision."
    
    confidence_value = abs(prob - 0.5)
    confidence = "High" if confidence_value > 0.3 else "Medium" if confidence_value > 0.15 else "Low"
    
    results_data = [
        ['Metric', 'Value'],
        ['Pneumonia Probability', f'{prob*100:.2f}%'],
        ['Normal Probability', f'{(1-prob)*100:.2f}%'],
        ['Decision Threshold', f'{threshold*100:.0f}%'],
        ['Final Diagnosis', diagnosis],
        ['AI Confidence Level', confidence]
    ]
    
    results_table = Table(results_data, colWidths=[3*inch, 3*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 4), (-1, 4), diagnosis_color),
        ('TEXTCOLOR', (0, 4), (-1, 4), colors.white),
        ('FONTNAME', (0, 4), (-1, 4), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    
    elements.append(results_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Clinical Interpretation
    elements.append(Paragraph("CLINICAL INTERPRETATION", heading_style))
    interpretation_text = f"<b>Diagnosis:</b> {diagnosis}<br/><br/><b>Recommendation:</b> {recommendation}"
    elements.append(Paragraph(interpretation_text, normal_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Images
    elements.append(Paragraph("RADIOLOGICAL IMAGES", heading_style))
    
    try:
        img_buffer = io.BytesIO()
        raw_img_resized = raw_img.resize((300, 300))
        raw_img_resized.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        original_img = RLImage(img_buffer, width=2.5*inch, height=2.5*inch)
        
        if heatmap_img is not None:
            heatmap_buffer = io.BytesIO()
            heatmap_pil = Image.fromarray(heatmap_img.astype('uint8'))
            heatmap_pil = heatmap_pil.resize((300, 300))
            heatmap_pil.save(heatmap_buffer, format='PNG')
            heatmap_buffer.seek(0)
            heatmap_rl = RLImage(heatmap_buffer, width=2.5*inch, height=2.5*inch)
            
            image_data = [
                ['Original X-Ray', 'Grad-CAM Heatmap'],
                [original_img, heatmap_rl]
            ]
            image_table = Table(image_data, colWidths=[3*inch, 3*inch])
        else:
            image_data = [['Original X-Ray'], [original_img]]
            image_table = Table(image_data, colWidths=[6*inch])
        
        image_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        
        elements.append(image_table)
    except Exception:
        elements.append(Paragraph("<i>Images could not be embedded</i>", normal_style))
    
    elements.append(Spacer(1, 0.3*inch))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=9,
                                     textColor=colors.HexColor('#666666'), alignment=TA_JUSTIFY,
                                     borderWidth=1, borderColor=colors.HexColor('#ffc107'),
                                     borderPadding=10, backColor=colors.HexColor('#fffbea'))
    
    disclaimer_text = """<b>⚠️ MEDICAL DISCLAIMER:</b> This report is generated by an AI diagnostic 
    assistant for screening purposes only. All findings must be reviewed by a qualified radiologist. 
    Not FDA-approved. Not for standalone clinical diagnosis."""
    
    elements.append(Paragraph(disclaimer_text, disclaimer_style))
    
    # Footer
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8,
                                 textColor=colors.grey, alignment=TA_CENTER)
    footer_text = f"Report ID: {report_id} | Generated: {report_date}"
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph(footer_text, footer_style))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ========================================================================
# SIDEBAR
# ========================================================================
st.sidebar.markdown("## 🏥 Clinical Controls")
st.sidebar.markdown("---")

if model is not None:
    st.sidebar.success("✅ Model Loaded")
else:
    st.sidebar.error("❌ Model Not Found")

threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)

if threshold < 0.3:
    st.sidebar.warning("⚠️ Aggressive Mode")
elif threshold > 0.7:
    st.sidebar.warning("⚠️ Conservative Mode")
else:
    st.sidebar.success("✓ Balanced Mode")

# ========================================================================
# MAIN HEADER
# ========================================================================
st.markdown("<h1 style='text-align: center;'>🫁 Smart Radiology: Pneumonia Diagnostic Workspace</h1>", 
            unsafe_allow_html=True)
st.markdown("---")

# ========================================================================
# MAIN APP
# ========================================================================
if model is None:
    st.error(f"⚠️ Model Loading Failed: {model_error}")
    st.stop()

uploaded_file = st.file_uploader("📂 Upload Chest X-Ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        raw_img = Image.open(uploaded_file).convert('RGB')
        input_size = (224, 224)
        processed_img = raw_img.resize(input_size)
        img_array = np.expand_dims(np.array(processed_img) / 255.0, axis=0)
        
        with st.spinner('🔍 Analyzing...'):
            prob = model.predict(img_array, verbose=0)[0][0]
            heatmap = get_heatmap(img_array, model)
        
        # Display results
        col1, col2, col3 = st.columns([1.5, 1.5, 1])
        
        with col1:
            st.markdown("### 📸 Original X-Ray")
            st.image(raw_img, use_container_width=True)
        
        with col2:
            st.markdown("### 🔍 AI Visual Evidence")
            if heatmap is not None:
                overlay_img = overlay_heatmap(processed_img, heatmap)
                st.image(overlay_img, use_container_width=True)
            else:
                st.image(processed_img, use_container_width=True)
        
        with col3:
            st.markdown("### 📋 Clinical Summary")
            st.metric("Pneumonia Probability", f"{prob*100:.1f}%")
            
            if prob >= threshold:
                st.error("🚨 PNEUMONIA DETECTED")
                st.caption("Urgent medical review recommended")
            elif prob < 0.25:
                st.success("✅ NORMAL / HEALTHY")
                st.caption("Standard follow-up")
            else:
                st.warning("⚠️ INCONCLUSIVE")
                st.caption("Manual review required")
        
        # Detailed metrics
        st.markdown("---")
        st.markdown("### 📊 Detailed Analysis")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pneumonia Score", f"{prob*100:.2f}%")
        m2.metric("Normal Score", f"{(1-prob)*100:.2f}%")
        m3.metric("Diagnosis", "Pneumonia" if prob >= threshold else "Normal" if prob < 0.25 else "Inconclusive")
        confidence_value = abs(prob - 0.5)
        confidence = "High" if confidence_value > 0.3 else "Medium" if confidence_value > 0.15 else "Low"
        m4.metric("AI Confidence", confidence)
        
        # Export section
        st.markdown("---")
        st.markdown("### 📄 Export Report")
        
        col_report1, col_report2 = st.columns(2)
        
        with col_report1:
            # Generate PDF
            overlay_img_array = overlay_heatmap(processed_img, heatmap) if heatmap is not None else None
            pdf_buffer = generate_pdf_report(uploaded_file, raw_img, prob, threshold, overlay_img_array)
            
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_buffer,
                file_name=f"pneumonia_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        
        with col_report2:
            csv_data = f"""Parameter,Value
Filename,{uploaded_file.name}
Date,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Pneumonia_Probability,{prob*100:.2f}%
Normal_Probability,{(1-prob)*100:.2f}%
Diagnosis,{"Pneumonia" if prob >= threshold else "Normal" if prob < 0.25 else "Inconclusive"}
Threshold,{threshold*100:.0f}%
Confidence,{confidence}
"""
            st.download_button(
                label="📊 Download CSV Data",
                data=csv_data,
                file_name=f"pneumonia_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

else:
    st.info("👆 Upload a chest X-ray image to begin analysis")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><b>⚠️ Medical Disclaimer:</b> AI screening tool only. All findings must be reviewed by qualified healthcare professionals.</p>
    <p>Powered by TensorFlow & Streamlit | Version 2.1</p>
</div>
""", unsafe_allow_html=True)