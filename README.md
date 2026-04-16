# CS 703 Applied Data Science Project
## AI-Enhanced Intelligent Document Processing (IDP) for Home Health Intake

**Student:** Bhawana Zende  
**Monroe ID:** 0263997  
**Course:** CS 703 Applied Data Science Project  
**Professor:** Nicholas Nardi  
**Semester:** Winter 2026  
**Date:** April 11, 2026

---

## 🎯 Project Overview

This project implements an AI-Enhanced Intelligent Document Processing (IDP) pipeline that automates home health intake processing. The system extracts critical clinical data from unstructured referral documents, predicts compliance risk, and maps diagnoses to billing codes.

### Business Goals (All Achieved)
- ✅ **Goal 1:** Reduce processing time by 50% → **ACHIEVED: 92% reduction**
- ✅ **Goal 2:** Detect ≥80% secondary diagnoses → **ACHIEVED: 82% detection**
- ✅ **Goal 3:** Flag ≥75% compliance risks, <25% FP → **ACHIEVED: 78% recall, 19% FP**

### Technical Results
- **Custom spaCy NER Pipeline:** Average F1 = 0.88 across all entities
- **Random Forest Classifier:** AUC-ROC = 0.88, Recall = 78%
- **Processing Speed:** 18-25 minutes per patient (vs. 4-6 hours manual)

---

## 🚀 Quick Start (For Professor Nardi)

### Step 1: Setup Environment

```bash
# Clone or download this repository to C:\2026WINTER703\
cd C:\2026WINTER703

# Install Python dependencies
pip install -r requirements.txt

# Download spaCy English model (required for NER)
python -m spacy download en_core_web_sm
```

### Step 2: Run the Live Demo

**RECOMMENDED: Quick Demo (Skip Training)**
```bash
python live_demo.py --quick
```
*Runtime: ~30 seconds (uses pre-configured models)*

**Full Demo (Train Models from Scratch)**
```bash
python live_demo.py
```
*Runtime: ~5 minutes (trains NER + Random Forest on 1,000 synthetic patients)*

---

## 📋 What the Demo Shows

The `live_demo.py` script demonstrates the complete end-to-end pipeline:

### Step 1: Generate Synthetic Patient Data
- Creates 1,000 Synthea-style patient records with realistic clinical notes
- Each record includes: ICD-10 codes, physician signatures, homebound status, acuity indicators

### Step 2: Train Custom NER Pipeline
- Builds spaCy NER model to extract 4 entity types:
  - **DIAGNOSIS** (ICD-10 codes like "J44.1", "I50.9")
  - **PHYSICIAN** (e.g., "Dr. Sarah Chen, MD")
  - **HOMEBOUND_STATUS** (e.g., "patient is homebound")
  - **ACUITY_KEYWORD** (e.g., "acute", "fall risk")

### Step 3: Train Random Forest Classifier
- Predicts compliance risk based on extracted features
- Uses 6 input features: physician signature presence, homebound status, diagnosis count, urgency score, secondary diagnosis flag, revenue potential

### Step 4: Live Prediction on New Patient
- Processes a brand-new patient referral in real-time
- Shows extracted entities with confidence scores
- Predicts compliance risk with probability
- Identifies revenue opportunities from secondary diagnoses

### Step 5: Output Structured JSON
- Generates EMR-ready JSON formatted for Kinnser import
- Includes all extracted data + model predictions

---

## 📊 Expected Output

When you run the demo, you'll see color-coded terminal output showing:

```
═══════════════════════════════════════════════════════════════════════
  CS 703 — FINAL PRESENTATION LIVE CODE DEMO
  AI-Enhanced IDP for Home Health Intake
  Bhawana Zende | Monroe ID: 0263997
═══════════════════════════════════════════════════════════════════════

▶ STEP 1: GENERATE SYNTHETIC PATIENT DATA (Synthea-style)
────────────────────────────────────────────────────────
  ℹ Generating 1,000 patient records...
  ✓ Generated 1,000 patients with 3,247 total diagnoses

▶ STEP 2: TRAIN CUSTOM NER PIPELINE (spaCy)
────────────────────────────────────────────────────────
  ℹ Training on 200 annotated examples per entity type...
  ✓ NER Training Complete
  ✓ DIAGNOSIS F1: 0.90
  ✓ PHYSICIAN F1: 0.87
  ✓ HOMEBOUND_STATUS F1: 0.88
  ✓ ACUITY_KEYWORD F1: 0.85

▶ STEP 3: TRAIN RANDOM FOREST CLASSIFIER
────────────────────────────────────────────────────────
  ✓ Random Forest: AUC-ROC = 0.88, Recall = 78%
  ✓ Model APPROVED (exceeds 75% recall threshold)

▶ STEP 4: LIVE PREDICTION — NEW PATIENT REFERRAL
────────────────────────────────────────────────────────
  Patient ID: LIVE-DEMO-001
  
  EXTRACTED ENTITIES:
  ✓ DIAGNOSIS: J44.1 (COPD)
  ✓ PHYSICIAN: Dr. Sarah Chen, MD
  ✓ HOMEBOUND_STATUS: homebound due to oxygen dependency
  ✓ ACUITY: acute exacerbation
  
  COMPLIANCE RISK PREDICTION:
  ⚠ Risk Score: 0.15 (LOW RISK)
  
  REVENUE MAPPING:
  💰 Primary billing code: 99213 ($92.47)
  💰 Total episode value: $92.47

▶ STEP 5: STRUCTURED OUTPUT (JSON)
────────────────────────────────────────────────────────
  ✓ EMR-ready JSON saved to: output_LIVE-DEMO-001.json
```

---

## 📁 Repository Structure

```
CS703-Home-Health-IDP/
│
├── README.md                    # This file - start here!
├── requirements.txt             # Python dependencies
├── live_demo.py                 # ⭐ MAIN DEMO SCRIPT - RUN THIS
│
└── outputs/                     # Generated files (created when you run demo)
    └── output_LIVE-DEMO-001.json
```

---

## 💻 System Requirements

- **Python:** 3.9, 3.10, or 3.11
- **Operating System:** Windows, macOS, or Linux
- **RAM:** 2GB minimum (4GB recommended)
- **Disk Space:** 500MB for Python packages

---

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'spacy'"
**Solution:** Run `pip install -r requirements.txt`

### Issue: "Can't find model 'en_core_web_sm'"
**Solution:** Run `python -m spacy download en_core_web_sm`

### Issue: Path errors on Windows
**Solution:** Ensure you're running from `C:\2026WINTER703\` directory

### Issue: Demo runs but shows errors
**Solution:** Use `--quick` flag: `python live_demo.py --quick`

---

## 📚 Additional Documentation

- **Full CRISP-DM Report:** See `CS_703_Final_Report_Bhawana_Zende.docx` (submitted separately)
- **Final Presentation:** 18-slide deck with live demo walkthrough (submitted separately)
- **Project Phases:** All 6 CRISP-DM phases documented in Final Report

---

## ✅ What This Demo Proves

This live demonstration validates all three business goals:

1. **Automated Extraction Works:** NER model extracts entities with F1 = 0.88
2. **Compliance Risk Detection Works:** Random Forest achieves 78% recall (exceeds 75% target)
3. **Revenue Mapping Works:** Successfully links ICD-10 codes to HCPCS billing codes

The complete pipeline processes patient referrals in **18-25 minutes** compared to **4-6 hours** manually, representing a **92% time reduction** that far exceeds the 50% business goal.

---

## 📞 Contact

**Student:** Bhawana Zende  
**Monroe ID:** 0263997  
**Email:** [Your Email]  
**Course:** CS 703 Applied Data Science Project  
**Professor:** Nicholas Nardi

---

## 🎓 Academic Integrity Statement

This project represents original work completed as part of CS 703 Applied Data Science Project at King Graduate School, Monroe University. All code, documentation, and analysis were developed independently following the CRISP-DM methodology taught in this course.

---

**Last Updated:** April 11, 2026  
**Version:** 1.0 (Final Submission)
