"""
PDF Report Generator for Pneumonia Detection Dashboard
Add this code to your app.py to generate professional PDF reports
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from datetime import datetime
import io
from PIL import Image

def generate_pdf_report(uploaded_file, raw_img, prob, threshold, heatmap_img=None):
    """
    Generate a professional PDF report for pneumonia diagnosis
    
    Parameters:
    - uploaded_file: Streamlit uploaded file object
    - raw_img: PIL Image object of the original X-ray
    - prob: Prediction probability (0-1)
    - threshold: Decision threshold (0-1)
    - heatmap_img: Optional numpy array of the Grad-CAM overlay
    
    Returns:
    - BytesIO buffer containing the PDF
    """
    
    # Create PDF buffer
    buffer = io.BytesIO()
    
    # Create document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    # Container for PDF elements
    elements = []
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        alignment=TA_JUSTIFY
    )
    
    # ========================================================================
    # 1. HEADER SECTION
    # ========================================================================
    
    # Logo placeholder (you can add your own logo)
    # logo = RLImage('path/to/logo.png', width=1.5*inch, height=1.5*inch)
    # elements.append(logo)
    
    # Title
    title = Paragraph("🫁 PNEUMONIA AI DIAGNOSTIC REPORT", title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.3*inch))
    
    # Report metadata
    report_date = datetime.now().strftime('%B %d, %Y at %I:%M %p')
    report_id = datetime.now().strftime('%Y%m%d%H%M%S')
    
    metadata = [
        ['Report ID:', report_id],
        ['Report Date:', report_date],
        ['Patient File:', uploaded_file.name],
        ['Analysis Model:', 'DenseNet121 Clinical v2.1'],
        ['Model Version:', '2.1 Clinical']
    ]
    
    metadata_table = Table(metadata, colWidths=[2*inch, 4*inch])
    metadata_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f2f6')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    
    elements.append(metadata_table)
    elements.append(Spacer(1, 0.4*inch))
    
    # ========================================================================
    # 2. DIAGNOSTIC RESULTS SECTION
    # ========================================================================
    
    elements.append(Paragraph("DIAGNOSTIC RESULTS", heading_style))
    
    # Determine diagnosis
    if prob >= threshold:
        diagnosis = "PNEUMONIA DETECTED"
        diagnosis_color = colors.HexColor('#dc3545')
        recommendation = "URGENT: Refer for immediate antibiotic screening and lung auscultation. Clinical review recommended within 24 hours."
    elif prob < 0.25:
        diagnosis = "NORMAL / HEALTHY"
        diagnosis_color = colors.HexColor('#28a745')
        recommendation = "Standard follow-up protocol. No lung opacities detected. Continue routine monitoring."
    else:
        diagnosis = "INCONCLUSIVE / BORDERLINE"
        diagnosis_color = colors.HexColor('#ffc107')
        recommendation = "AI confidence is moderate. Manual radiologist review required before clinical decision. Consider additional imaging if symptoms present."
    
    # Calculate confidence level
    confidence_value = abs(prob - 0.5)
    if confidence_value > 0.3:
        confidence = "High"
    elif confidence_value > 0.15:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    # Results table
    results_data = [
        ['Metric', 'Value', 'Interpretation'],
        ['Pneumonia Probability', f'{prob*100:.2f}%', ''],
        ['Normal Probability', f'{(1-prob)*100:.2f}%', ''],
        ['Decision Threshold', f'{threshold*100:.0f}%', ''],
        ['Final Diagnosis', diagnosis, ''],
        ['AI Confidence Level', confidence, '']
    ]
    
    results_table = Table(results_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
    results_table.setStyle(TableStyle([
        # Header row
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        
        # Data rows
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        
        # Diagnosis row highlight
        ('BACKGROUND', (0, 4), (-1, 4), diagnosis_color),
        ('TEXTCOLOR', (0, 4), (-1, 4), colors.white),
        ('FONTNAME', (0, 4), (-1, 4), 'Helvetica-Bold'),
        
        # Borders
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
    ]))
    
    elements.append(results_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # ========================================================================
    # 3. CLINICAL INTERPRETATION
    # ========================================================================
    
    elements.append(Paragraph("CLINICAL INTERPRETATION", heading_style))
    
    interpretation_text = f"""
    <b>Diagnosis:</b> {diagnosis}<br/>
    <br/>
    <b>Analysis:</b> The AI model has analyzed the chest X-ray with a pneumonia probability 
    score of {prob*100:.2f}%. Based on the configured decision threshold of {threshold*100:.0f}%, 
    the system has classified this case as <b>{diagnosis}</b> with <b>{confidence.lower()}</b> confidence.<br/>
    <br/>
    <b>Clinical Recommendation:</b> {recommendation}
    """
    
    elements.append(Paragraph(interpretation_text, normal_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # ========================================================================
    # 4. IMAGES SECTION
    # ========================================================================
    
    elements.append(Paragraph("RADIOLOGICAL IMAGES", heading_style))
    
    # Save images to temporary buffers
    # Original X-ray
    img_buffer = io.BytesIO()
    raw_img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    # Add original image
    try:
        original_img = RLImage(img_buffer, width=3*inch, height=3*inch)
        
        # Create image table
        if heatmap_img is not None:
            # Save heatmap
            heatmap_buffer = io.BytesIO()
            heatmap_pil = Image.fromarray(heatmap_img.astype('uint8'))
            heatmap_pil.save(heatmap_buffer, format='PNG')
            heatmap_buffer.seek(0)
            
            heatmap_rl = RLImage(heatmap_buffer, width=3*inch, height=3*inch)
            
            image_data = [
                ['Original X-Ray', 'AI Visual Evidence (Grad-CAM)'],
                [original_img, heatmap_rl],
                ['Uploaded chest X-ray image', 'Red areas show AI focus regions']
            ]
            
            image_table = Table(image_data, colWidths=[3.25*inch, 3.25*inch])
            image_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('FONTNAME', (0, 2), (-1, 2), 'Helvetica'),
                ('FONTSIZE', (0, 2), (-1, 2), 9),
                ('TEXTCOLOR', (0, 2), (-1, 2), colors.grey),
                ('GRID', (0, 0), (-1, 1), 0.5, colors.grey),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
            ]))
        else:
            image_data = [
                ['Original X-Ray'],
                [original_img],
                ['Uploaded chest X-ray image']
            ]
            
            image_table = Table(image_data, colWidths=[6.5*inch])
            image_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('GRID', (0, 0), (-1, 1), 0.5, colors.grey),
            ]))
        
        elements.append(image_table)
    except Exception as e:
        elements.append(Paragraph(f"<i>Images could not be embedded: {str(e)}</i>", normal_style))
    
    elements.append(Spacer(1, 0.3*inch))
    
    # ========================================================================
    # 5. TECHNICAL DETAILS
    # ========================================================================
    
    elements.append(Paragraph("TECHNICAL DETAILS", heading_style))
    
    technical_text = """
    <b>Model Architecture:</b> DenseNet121 with transfer learning<br/>
    <b>Input Preprocessing:</b> Image resized to 224×224 pixels, normalized to [0,1] range<br/>
    <b>Explainability Method:</b> Grad-CAM (Gradient-weighted Class Activation Mapping)<br/>
    <b>Target Performance:</b> Sensitivity >95%, Specificity >90%<br/>
    <b>Training Dataset:</b> Kaggle Chest X-Ray Pneumonia Dataset (5,863 images)<br/>
    """
    
    elements.append(Paragraph(technical_text, normal_style))
    elements.append(Spacer(1, 0.4*inch))
    
    # ========================================================================
    # 6. DISCLAIMER
    # ========================================================================
    
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=9,
        leading=11,
        textColor=colors.HexColor('#666666'),
        alignment=TA_JUSTIFY,
        borderWidth=1,
        borderColor=colors.HexColor('#ffc107'),
        borderPadding=10,
        backColor=colors.HexColor('#fffbea')
    )
    
    disclaimer_text = """
    <b>⚠️ IMPORTANT MEDICAL DISCLAIMER:</b><br/>
    <br/>
    This report is generated by an artificial intelligence diagnostic assistant and is intended 
    for <b>screening purposes only</b>. This AI system is NOT FDA-approved and should NOT be used 
    for standalone clinical diagnosis.<br/>
    <br/>
    <b>All findings must be reviewed and confirmed by a qualified healthcare professional, 
    preferably a board-certified radiologist.</b> The AI's predictions should be considered as 
    a supplementary tool to assist in clinical decision-making, not as a replacement for 
    professional medical judgment.<br/>
    <br/>
    This system has not undergone clinical validation in a real-world hospital setting. 
    Performance may vary depending on image quality, patient demographics, and disease presentation. 
    False negatives and false positives are possible.<br/>
    <br/>
    For urgent medical concerns, please consult with a physician immediately.
    """
    
    elements.append(Paragraph(disclaimer_text, disclaimer_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # ========================================================================
    # 7. FOOTER
    # ========================================================================
    
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=TA_CENTER
    )
    
    footer_text = f"""
    This report was automatically generated by the Pneumonia AI Diagnostic System v2.1<br/>
    Powered by TensorFlow and Streamlit | Report ID: {report_id}<br/>
    Generated on {report_date}
    """
    
    elements.append(Paragraph(footer_text, footer_style))
    
    # ========================================================================
    # BUILD PDF
    # ========================================================================
    
    doc.build(elements)
    buffer.seek(0)
    
    return buffer


# ============================================================================
# INTEGRATION CODE FOR YOUR STREAMLIT APP
# ============================================================================

"""
ADD THIS TO YOUR app.py IN THE DOWNLOAD SECTION:

Replace the download button section with this code:
"""

# Example integration in Streamlit:
"""
col_report1, col_report2 = st.columns(2)

with col_report1:
    # Generate PDF report
    if heatmap is not None:
        overlay_img_array = overlay_heatmap(processed_img, heatmap)
    else:
        overlay_img_array = None
    
    pdf_buffer = generate_pdf_report(
        uploaded_file=uploaded_file,
        raw_img=raw_img,
        prob=prob,
        threshold=threshold,
        heatmap_img=overlay_img_array
    )
    
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_buffer,
        file_name=f"pneumonia_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

with col_report2:
    # CSV data (keep existing)
    csv_data = f'''Parameter,Value
Filename,{uploaded_file.name}
Date,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Pneumonia_Probability,{prob*100:.2f}%
Normal_Probability,{(1-prob)*100:.2f}%
Diagnosis,{"Pneumonia" if prob >= threshold else "Normal" if prob < 0.25 else "Inconclusive"}
Threshold,{threshold*100:.0f}%
'''
    
    st.download_button(
        label="📊 Download CSV Data",
        data=csv_data,
        file_name=f"pneumonia_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
"""