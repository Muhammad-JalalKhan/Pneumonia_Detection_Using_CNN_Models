# 🫁 Pneumonia Detection Using CNN Models

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![TensorFlow 2.16+](https://img.shields.io/badge/TensorFlow-2.16%2B-orange)](https://www.tensorflow.org/)

An advanced AI-powered diagnostic system for pneumonia detection using chest X-ray images. This project leverages deep learning (CNN models) to provide clinical-grade pneumonia classification with explainability through Grad-CAM visualization and comprehensive reporting capabilities.

---

## 📋 Table of Contents

- [Features](#features)
- [Technical Specifications](#technical-specifications)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Output & Reports](#output--reports)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)
- [Medical Disclaimer](#-medical-disclaimer)

---

## ✨ Features

### 🎯 Core Functionality
- **Pneumonia Detection**: Binary classification (Pneumonia vs Normal) with >95% target sensitivity
- **Real-time Analysis**: Process chest X-ray images instantly
- **Clinical Scoring**: Probability scores with confidence levels
- **Grad-CAM Visualization**: Visual explanation of AI decision-making with heatmap overlay

### 📊 Reporting & Export
- **PDF Reports**: Professional, clinical-grade PDF generation with metadata
- **CSV Export**: Structured data export for further analysis
- **Text Reports**: Plain-text summaries for quick reference
- **Metadata Tracking**: Report IDs, timestamps, and analysis parameters

### 🏥 Clinical Features
- **Adjustable Decision Threshold**: Customize sensitivity/specificity trade-off
- **Confidence Levels**: High/Medium/Low confidence indicators
- **Clinical Recommendations**: Automated guidance based on diagnosis
- **Multi-image Support**: Batch analysis capabilities
- **Error Handling**: Robust model loading with fallback strategies

### 🖥️ User Interfaces
- **Streamlit Dashboard** (`dashboard.py`): Professional web UI with real-time visualization
- **Flask Apps** (`app.py`, `appwithPDF.py`): RESTful API alternatives
- **PDF Generation** (`pdf_generator.py`): Standalone report generator

---

## 🔬 Technical Specifications

| Component | Details |
|-----------|---------|
| **Architecture** | DenseNet121 Clinical v2.1 |
| **Input Size** | 224×224 RGB images |
| **Output** | Binary probability (0-1) |
| **Target Sensitivity** | >95% |
| **Explainability** | Grad-CAM heatmaps |
| **Model Format** | Keras H5 & .keras files |
| **Framework** | TensorFlow 2.16+ |
| **UI Framework** | Streamlit 1.28+ |

---
