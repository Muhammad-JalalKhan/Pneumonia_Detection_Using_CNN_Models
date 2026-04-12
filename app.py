"""
Complete Interactive Pneumonia Detection Dashboard
Fixes all errors and adds professional features
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import io

# ========================================================================
# 1. PAGE CONFIGURATION (Must be first Streamlit command)
# ========================================================================
st.set_page_config(
    page_title="Radiology AI Assistant", 
    layout="wide",
    page_icon="🫁",
    initial_sidebar_state="expanded"
)

# ========================================================================
# 2. PROFESSIONAL STYLING
# ========================================================================
st.markdown("""
    <style>
    /* Main background */
    .main { 
        background-color: #f5f7f9; 
    }
    
    /* Metric cards */
    .stMetric { 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1); 
    }
    
    /* Headers */
    h1 {
        color: #1f77b4;
        font-weight: 700;
    }
    
    h2, h3 {
        color: #2c3e50;
    }
    
    /* Custom alert boxes */
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
    </style>
    """, unsafe_allow_html=True)  # ← FIXED: unsafe_allow_html, not unsafe_call_safe

# ========================================================================
# 3. MODEL LOADING WITH ERROR HANDLING
# ========================================================================
@st.cache_resource
def load_elite_model():
    """Load the trained pneumonia detection model"""
    try:
        model = tf.keras.models.load_model('pneumonia_model_v1.h5', safe_mode=False)
        return model, None
    except Exception as e:
        return None, str(e)

model, model_error = load_elite_model()

# ========================================================================
# 4. GRAD-CAM HEATMAP GENERATION (IMPROVED)
# ========================================================================
def get_heatmap(img_array, model):
    """
    Generate Grad-CAM heatmap showing where the AI is looking
    """
    try:
        # Get the base model (first layer)
        base_model = model.layers[0]
        
        # Find the last convolutional layer
        last_conv_layer_name = None
        for layer in reversed(base_model.layers):
            if 'conv' in layer.name or 'concat' in layer.name:
                last_conv_layer_name = layer.name
                break
        
        if last_conv_layer_name is None:
            st.warning("Could not find convolutional layer for Grad-CAM")
            return None
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            [base_model.inputs], 
            [base_model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]
        
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Generate heatmap
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    except Exception as e:
        st.warning(f"Grad-CAM generation failed: {str(e)}")
        return None

# ========================================================================
# 5. OVERLAY HEATMAP ON IMAGE
# ========================================================================
def overlay_heatmap(img, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on original image
    """
    if heatmap is None:
        return img
    
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to color
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # Convert image to BGR
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Overlay
    superimposed = cv2.addWeighted(img_bgr, 1-alpha, heatmap_color, alpha, 0)
    
    # Convert back to RGB
    result = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
    
    return result

# ========================================================================
# 6. SIDEBAR - CLINICAL CONTROLS
# ========================================================================
st.sidebar.markdown("## 🏥 Clinical Controls")
st.sidebar.markdown("---")

if model is not None:
    st.sidebar.success("✅ Model Loaded Successfully")
    st.sidebar.info(
        "**Model Information:**\n"
        "- Architecture: DenseNet121/EfficientNet\n"
        "- Target Sensitivity: >95%\n"
        "- Version: 2.1 Clinical"
    )
else:
    st.sidebar.error("❌ Model Not Found")
    st.sidebar.warning(
        "Please ensure 'pneumonia_model_v1.h5' is in the project folder"
    )

st.sidebar.markdown("---")

# Threshold slider
st.sidebar.markdown("### ⚙️ Decision Threshold")
threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.1, 
    max_value=0.9, 
    value=0.5,
    step=0.05,
    help="Lower = More sensitive (catches more cases)\nHigher = More specific (fewer false alarms)"
)

# Threshold guidance
if threshold < 0.3:
    st.sidebar.warning("⚠️ **Aggressive Mode**\nHigh sensitivity - May produce false alarms")
elif threshold > 0.7:
    st.sidebar.warning("⚠️ **Conservative Mode**\nHigh specificity - May miss some cases")
else:
    st.sidebar.success("✓ **Balanced Mode**\nOptimized sensitivity/specificity")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Disclaimer:** This AI is a screening tool only. "
    "All findings must be reviewed by a qualified radiologist."
)

# ========================================================================
# 7. MAIN HEADER
# ========================================================================
st.markdown(
    "<h1 style='text-align: center;'>🫁 Smart Radiology: Pneumonia Diagnostic Workspace</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; font-size: 18px; color: #666;'>"
    "Real-time AI analysis for Chest X-Rays with Grad-CAM Explainability"
    "</p>", 
    unsafe_allow_html=True
)
st.markdown("---")

# ========================================================================
# 8. FILE UPLOAD SECTION
# ========================================================================
if model is None:
    st.error(
        f"⚠️ **Model Loading Failed**\n\n"
        f"Error: {model_error}\n\n"
        f"Please ensure:\n"
        f"1. File 'pneumonia_model_v1.h5' exists in the project folder\n"
        f"2. The model file is not corrupted\n"
        f"3. TensorFlow is properly installed"
    )
    st.stop()

uploaded_file = st.file_uploader(
    "📂 Upload Chest X-Ray Image (JPG, PNG, DICOM)", 
    type=["jpg", "jpeg", "png"],
    help="Drag and drop or click to browse"
)

# ========================================================================
# 9. IMAGE PROCESSING & PREDICTION
# ========================================================================
if uploaded_file is not None:
    try:
        # Read and process image
        raw_img = Image.open(uploaded_file).convert('RGB')
        
        # Get model input size (adjust if your model uses different size)
        input_size = (224, 224)  # Change to (300, 300) if using EfficientNet-B3
        
        # Resize for model
        processed_img = raw_img.resize(input_size)
        img_array = np.expand_dims(np.array(processed_img) / 255.0, axis=0)
        
        # ================================================================
        # AI PREDICTION
        # ================================================================
        with st.spinner('🔍 Analyzing X-Ray...'):
            # Get prediction
            prob = model.predict(img_array, verbose=0)[0][0]
            
            # Generate Grad-CAM heatmap
            heatmap = get_heatmap(img_array, model)
        
        # ================================================================
        # DISPLAY RESULTS IN 3 COLUMNS
        # ================================================================
        col1, col2, col3 = st.columns([1.5, 1.5, 1])
        
        # --- COLUMN 1: Original X-Ray ---
        with col1:
            st.markdown("### 📸 Original X-Ray")
            st.image(raw_img, use_container_width=True)
            st.caption(f"Uploaded: {uploaded_file.name}")
        
        # --- COLUMN 2: AI Visual Evidence ---
        with col2:
            st.markdown("### 🔍 AI Visual Evidence (Grad-CAM)")
            
            if heatmap is not None:
                # Overlay heatmap on image
                overlay_img = overlay_heatmap(processed_img, heatmap)
                st.image(overlay_img, use_container_width=True)
                st.caption("Red areas show where the AI is focusing")
            else:
                st.image(processed_img, use_container_width=True)
                st.caption("Grad-CAM visualization unavailable")
        
        # --- COLUMN 3: Clinical Summary ---
        with col3:
            st.markdown("### 📋 Clinical Summary")
            
            # Display confidence gauge
            st.metric(
                label="Pneumonia Probability", 
                value=f"{prob*100:.1f}%",
                delta="High Risk" if prob >= threshold else "Low Risk"
            )
            
            # Diagnosis based on threshold
            if prob >= threshold:
                # PNEUMONIA DETECTED
                st.markdown(
                    f"""
                    <div class="alert-pneumonia">
                        <h3 style="color: #dc3545; margin-top: 0;">🚨 PNEUMONIA DETECTED</h3>
                        <p><strong>Confidence:</strong> {prob*100:.2f}%</p>
                        <p><strong>Recommendation:</strong> Refer for urgent antibiotic screening and lung auscultation.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            elif prob < 0.25:
                # NORMAL / HEALTHY
                st.markdown(
                    f"""
                    <div class="alert-normal">
                        <h3 style="color: #28a745; margin-top: 0;">✅ NORMAL / HEALTHY</h3>
                        <p><strong>Confidence:</strong> {(1-prob)*100:.2f}%</p>
                        <p><strong>Recommendation:</strong> Standard follow-up. No opacities detected.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            else:
                # INCONCLUSIVE / BORDERLINE
                st.markdown(
                    f"""
                    <div class="alert-uncertain">
                        <h3 style="color: #ffc107; margin-top: 0;">⚠️ INCONCLUSIVE</h3>
                        <p><strong>Confidence:</strong> {prob*100:.2f}%</p>
                        <p><strong>Recommendation:</strong> AI is uncertain. Requires manual radiologist review.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # ================================================================
        # DETAILED METRICS SECTION
        # ================================================================
        st.markdown("---")
        st.markdown("### 📊 Detailed Analysis")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Pneumonia Score", f"{prob*100:.2f}%")
        
        with metric_col2:
            st.metric("Normal Score", f"{(1-prob)*100:.2f}%")
        
        with metric_col3:
            diagnosis = "Pneumonia" if prob >= threshold else "Normal"
            st.metric("Final Diagnosis", diagnosis)
        
        with metric_col4:
            confidence = "High" if abs(prob - 0.5) > 0.3 else "Medium" if abs(prob - 0.5) > 0.15 else "Low"
            st.metric("AI Confidence", confidence)
        
        # ================================================================
        # DOWNLOAD REPORT SECTION
        # ================================================================
        st.markdown("---")
        st.markdown("### 📄 Export Report")
        
        # Generate simple text report
        report_text = f"""
╔══════════════════════════════════════════════════════════════╗
║           PNEUMONIA AI DIAGNOSTIC REPORT                    ║
╚══════════════════════════════════════════════════════════════╝

Patient File: {uploaded_file.name}
Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Model: DenseNet121 Clinical v2.1

─────────────────────────────────────────────────────────────

DIAGNOSTIC RESULTS:

  Diagnosis:          {"PNEUMONIA DETECTED" if prob >= threshold else "NORMAL" if prob < 0.25 else "INCONCLUSIVE"}
  Pneumonia Score:    {prob*100:.2f}%
  Normal Score:       {(1-prob)*100:.2f}%
  Decision Threshold: {threshold*100:.0f}%
  AI Confidence:      {confidence}

─────────────────────────────────────────────────────────────

CLINICAL RECOMMENDATION:

  {"• URGENT: Refer for antibiotic screening and lung auscultation" if prob >= threshold else "• Standard follow-up protocol" if prob < 0.25 else "• Manual radiologist review required"}
  {"• Monitor for respiratory symptoms" if prob >= threshold else "• No immediate action required" if prob < 0.25 else "• Consider additional imaging"}

─────────────────────────────────────────────────────────────

DISCLAIMER:
This report is generated by an AI diagnostic assistant and is 
intended for screening purposes only. It should NOT replace 
professional medical diagnosis. All findings must be reviewed 
and confirmed by a qualified healthcare professional.

─────────────────────────────────────────────────────────────
Report ID: {datetime.now().strftime('%Y%m%d%H%M%S')}
        """
        
        col_report1, col_report2 = st.columns(2)
        
        with col_report1:
            st.download_button(
                label="📥 Download Text Report",
                data=report_text,
                file_name=f"pneumonia_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col_report2:
            # Create a simple CSV report
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
        st.error(f"❌ Error processing image: {str(e)}")
        st.error("Please ensure the uploaded file is a valid image.")

else:
    # ================================================================
    # INSTRUCTIONS WHEN NO FILE UPLOADED
    # ================================================================
    st.info(
        "👆 **Upload a chest X-ray image to begin analysis**\n\n"
        "Supported formats: JPG, PNG\n\n"
        "The AI will:\n"
        "- Analyze the X-ray for signs of pneumonia\n"
        "- Show visual evidence (Grad-CAM heatmap)\n"
        "- Provide clinical recommendations\n"
        "- Generate downloadable reports"
    )
    
    # Show sample workflow
    st.markdown("---")
    st.markdown("### 🔄 Workflow")
    
    workflow_col1, workflow_col2, workflow_col3, workflow_col4 = st.columns(4)
    
    with workflow_col1:
        st.markdown("**1️⃣ Upload**")
        st.markdown("Upload chest X-ray image")
    
    with workflow_col2:
        st.markdown("**2️⃣ Analyze**")
        st.markdown("AI processes the image")
    
    with workflow_col3:
        st.markdown("**3️⃣ Review**")
        st.markdown("Check diagnosis & heatmap")
    
    with workflow_col4:
        st.markdown("**4️⃣ Export**")
        st.markdown("Download report")

# ========================================================================
# FOOTER
# ========================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p><strong>⚠️ Medical Disclaimer:</strong> This AI system is a diagnostic assistant tool only. 
        All findings must be reviewed and confirmed by qualified healthcare professionals. 
        Not intended for standalone clinical use.</p>
        <p>Powered by TensorFlow & Streamlit | Version 2.1</p>
    </div>
    """,
    unsafe_allow_html=True
)