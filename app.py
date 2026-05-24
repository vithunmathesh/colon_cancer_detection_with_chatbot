import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pickle
import requests
import os
from pathlib import Path
from ultralytics import YOLO
import io
from dotenv import load_dotenv
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# Load environment variables from .env file
load_dotenv()

# Set page config
st.set_page_config(
    page_title="AI Colonoscopy Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for configuration
st.sidebar.title("⚙️ Configuration")

# Load API key from .env file securely
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.sidebar.error("❌ GOOGLE_API_KEY not found in .env file")
    st.sidebar.info("Please add your Google AI Studio API key to the .env file")
else:
    st.sidebar.success("✅ Google Gemini API Key Loaded")

st.sidebar.markdown("---")
confidence_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.3, 1.0, 0.5)

# Function to load YOLO model
@st.cache_resource
def load_yolo_model():
    try:
        model_path = Path("best.pt")
        if not model_path.exists():
            st.error("❌ best.pt not found. Please ensure the YOLO model file is in the current directory.")
            return None
        model = YOLO(str(model_path))
        return model
    except Exception as e:
        st.error(f"❌ Error loading YOLO model: {str(e)}")
        return None

# Function to load EfficientNet model for feature extraction
@st.cache_resource
def load_efficientnet_model():
    try:
        # Load EfficientNet-B0 pretrained on ImageNet
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # Remove the final classification layer to get features
        efficientnet = torch.nn.Sequential(*list(efficientnet.children())[:-1])
        efficientnet.eval()
        return efficientnet
    except Exception as e:
        st.error(f"❌ Error loading EfficientNet model: {str(e)}")
        return None

# Function to load RandomForest classifier model
@st.cache_resource
def load_classifier_model():
    try:
        model_path = Path("rf_cancer_model.pkl")
        if not model_path.exists():
            st.error("❌ rf_cancer_model.pkl not found. Please ensure the classifier model file is in the current directory.")
            return None
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"❌ Error loading classifier model: {str(e)}")
        return None

# Function to prepare image for YOLO detection
def prepare_image_for_yolo(image):
    """Convert PIL image to format suitable for YOLO"""
    return np.array(image)

# Function to draw bounding boxes on image
def draw_bounding_boxes(image, yolo_results):
    """Draw detected polyp bounding boxes on image"""
    try:
        img_array = np.array(image).copy()
        
        if yolo_results is None or len(yolo_results.boxes) == 0:
            return Image.fromarray(img_array)
        
        # Draw boxes for each detection
        boxes = yolo_results.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Draw red rectangle
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Draw confidence label
            label = f"Polyp {conf:.2%}"
            cv2.putText(
                img_array,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )
        
        return Image.fromarray(img_array)
    except Exception as e:
        st.warning(f"⚠️ Could not draw bounding boxes: {str(e)}")
        return image

def detect_polyp(yolo_model, image):
    """Detect polyp in the image using YOLO"""
    try:
        if yolo_model is None:
            return False, 0.0, None

        img_array = prepare_image_for_yolo(image)
        results = yolo_model(img_array, conf=confidence_threshold)

        if len(results) > 0 and len(results[0].boxes) > 0:
            # Get highest confidence detection
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            max_confidence = float(np.max(confidences))
            return True, max_confidence, results[0]

        return False, 0.0, None
    except Exception as e:
        st.error(f"❌ Error in polyp detection: {str(e)}")
        return False, 0.0, None

# Function to prepare image for classifier using EfficientNet features
def prepare_image_for_classifier(image, yolo_model):
    """Prepare image for classifier by extracting EfficientNet features from YOLO-detected region"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)

        # Handle different image formats
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

        # Use YOLO to detect polyp and get bounding box
        results = yolo_model(img_array, conf=0.25)  # Lower confidence for feature extraction

        if len(results) > 0 and len(results[0].boxes) > 0:
            # Get the first detection (highest confidence by default in YOLO)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            x1, y1, x2, y2 = map(int, boxes[0])  # Get first box

            # Crop the polyp region
            crop = img_array[y1:y2, x1:x2]

            if crop.size == 0:
                # If crop is empty, use full image
                crop = img_array
        else:
            # If no polyp detected, use full image
            crop = img_array

        # Define transforms (same as in training)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Apply transforms
        input_tensor = transform(crop).unsqueeze(0)

        return input_tensor
    except Exception as e:
        st.error(f"❌ Error preparing image for classifier: {str(e)}")
        return None

# Function to classify cancer risk using EfficientNet features + RandomForest
def classify_risk(yolo_model, efficientnet_model, classifier_model, image):
    """Classify cancer risk using EfficientNet feature extraction + RandomForest classifier"""
    try:
        if efficientnet_model is None or classifier_model is None:
            return "Unable to classify", 0.0, "Unknown"

        # Prepare image for feature extraction (gets tensor from EfficientNet preprocessing)
        img_tensor = prepare_image_for_classifier(image, yolo_model)
        if img_tensor is None:
            return "Error", 0.0, "Unknown"

        # Extract features using EfficientNet
        with torch.no_grad():
            features = efficientnet_model(img_tensor)
            # Flatten features to 1D vector
            features = features.squeeze().numpy().reshape(1, -1)

        # Handle both sklearn models and custom models
        try:
            if hasattr(classifier_model, 'predict_proba'):
                probs = classifier_model.predict_proba(features)[0]
                class_names = classifier_model.classes_

                # Create probability dictionary
                prob_dict = dict(zip(class_names, probs))

                # Get predicted class
                predicted_class = class_names[np.argmax(probs)]

                # Calculate high risk (Adenomatous + Serrated_Lesions)
                high_risk_prob = prob_dict.get("Adenomatous", 0) + prob_dict.get("Serrated_Lesions", 0)
                # Calculate low risk (Hyperplastic)
                low_risk_prob = prob_dict.get("Hyperplastic", 0)

                # Use high risk probability as the risk score
                risk_score = high_risk_prob

                # Categorize risk
                if risk_score >= 0.7:
                    risk_level = "High Risk"
                elif risk_score >= 0.4:
                    risk_level = "Medium Risk"
                else:
                    risk_level = "Low Risk"

                return risk_level, risk_score, predicted_class
            else:
                # Fallback to original method if predict_proba not available
                predictions = classifier_model.predict(features)
                risk_score = float(predictions[0]) if np.max(predictions) <= 1.0 else float(predictions[0]) / 100.0
                risk_score = max(0.0, min(1.0, risk_score))

                # Get predicted class (if available)
                if hasattr(classifier_model, 'classes_'):
                    predicted_class = classifier_model.classes_[0] if len(predictions) > 0 else "Unknown"
                else:
                    predicted_class = "Unknown"

                if risk_score >= 0.7:
                    risk_level = "High Risk"
                elif risk_score >= 0.4:
                    risk_level = "Medium Risk"
                else:
                    risk_level = "Low Risk"

                return risk_level, risk_score, predicted_class
        except Exception as model_error:
            st.error(f"❌ Model prediction error: {str(model_error)}")
            return "Prediction Error", 0.0, "Unknown"

    except Exception as e:
        st.error(f"❌ Error in risk classification: {str(e)}")
        return "Classification Error", 0.0, "Unknown"

# Function to generate report using Gemini API
def generate_report_with_gemini(api_key, polyp_detected, confidence, risk_level, risk_score, polyp_type):
    """Generate patient-friendly report using Google Gemini API"""
    try:
        if not api_key:
            st.warning("⚠️ No API key provided. Using default report format.")
            return generate_default_report(polyp_detected, risk_level, polyp_type)

        # Prepare the prompt
        if polyp_detected:
            prompt = f"""You are a medical report writer. Generate a patient-friendly medical report based on these findings:

- Polyp Status: DETECTED
- Polyp Type: {polyp_type}
- Detection Confidence: {confidence:.2%}
- Risk Classification: {risk_level}
- Risk Score: {risk_score:.2%}

Please generate:
1. A brief, easy-to-understand summary of findings
2. What this means for the patient
3. Recommended next steps
4. Disclaimer that this is AI-assisted analysis and consultation with a doctor is essential

Keep the tone professional but compassionate. Avoid medical jargon."""
        else:
            prompt = f"""You are a medical report writer. Generate a patient-friendly medical report based on these findings:

- Polyp Status: NOT DETECTED
- Scan Result: No abnormal growths observed

Please generate:
1. A brief summary of findings
2. What this positive result means
3. Recommended follow-up schedule
4. Reminder about importance of regular checkups
5. Disclaimer that this is AI-assisted analysis

Keep the tone professional but reassuring."""

        # Try different Gemini models with correct API endpoint
        models = ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-pro"]
        
        for model_name in models:
            try:
                url = f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent?key={api_key}"
                
                headers = {
                    "Content-Type": "application/json"
                }
                
                data = {
                    "contents": [
                        {
                            "parts": [
                                {
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.7,
                        "maxOutputTokens": 1000,
                    }
                }
                
                response = requests.post(
                    url,
                    json=data,
                    headers=headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract text from response
                    if 'candidates' in result and len(result['candidates']) > 0:
                        candidate = result['candidates'][0]
                        if 'content' in candidate and 'parts' in candidate['content']:
                            text = candidate['content']['parts'][0].get('text', '')
                            if text:
                                return text
                    
                    # If we got here, the response format was unexpected
                    continue
                    
                elif response.status_code == 400:
                    error_data = response.json()
                    continue
                    
                else:
                    continue
                    
            except requests.exceptions.Timeout:
                continue
            except Exception as e:
                continue
        
        # If all models failed, use default report silently
        return generate_default_report(polyp_detected, risk_level, polyp_type)
        
    except Exception as e:
        st.error(f"❌ Error generating report: {str(e)}")
        return generate_default_report(polyp_detected, risk_level, polyp_type)

def generate_default_report(polyp_detected, risk_level, polyp_type="Unknown"):
    """Generate default report if API fails"""
    if polyp_detected:
        return f"""📋 COLONOSCOPY ANALYSIS REPORT

FINDINGS:
A growth has been detected in the colon during the AI analysis.
Polyp Type: {polyp_type}

RISK ASSESSMENT:
The detected lesion has been classified as: {risk_level}

RECOMMENDATION:
This preliminary analysis suggests further medical consultation is needed. Please schedule an appointment with your gastroenterologist for:
- Confirmation of findings
- Additional diagnostic procedures if needed
- Discussion of treatment options

IMPORTANT DISCLAIMER:
This is an AI-assisted preliminary analysis only and should not be considered a definitive diagnosis. Only a qualified medical professional can make a formal diagnosis. This report is intended to support, not replace, professional medical judgment.

NEXT STEPS:
1. Consult with your healthcare provider
2. Bring this report to your appointment
3. Follow your doctor's recommendations"""
    else:
        return """📋 COLONOSCOPY ANALYSIS REPORT

FINDINGS:
No abnormal growths or polyps were detected in the colonoscopy images analyzed.

RESULT:
This is a positive finding and suggests the colon appears normal from the analyzed images.

RECOMMENDATION:
Continue with regular screening intervals as recommended by your healthcare provider.

FOLLOW-UP SCHEDULE:
- If this is your first normal colonoscopy: Follow-up in 10 years
- If you have risk factors: Follow-up as per your doctor's recommendation
- If you have a family history: Consult with your doctor for personalized screening schedule

IMPORTANT DISCLAIMER:
This is an AI-assisted preliminary analysis only and should not be considered a definitive diagnosis. Only a qualified medical professional can make a formal diagnosis. This report is intended to support, not replace, professional medical judgment.

LIFESTYLE RECOMMENDATIONS:
- Maintain a healthy diet rich in fiber
- Exercise regularly
- Avoid smoking and excessive alcohol
- Schedule regular health checkups"""

# Main UI
st.title("🔬 AI Colonoscopy Analysis System")
st.markdown("Powered by YOLO Detection + EfficientNet Classifier + Google Gemini Report Generation")

# Load models
with st.spinner("Loading AI models..."):
    yolo_model = load_yolo_model()
    efficientnet_model = load_efficientnet_model()
    classifier_model = load_classifier_model()

if yolo_model is None or efficientnet_model is None or classifier_model is None:
    st.error("❌ Unable to load required models. Please ensure best.pt and rf_cancer_model.pkl are in the current directory.")
    st.stop()

st.success("✅ Models loaded successfully!")

# File uploader
uploaded_file = st.file_uploader("📤 Upload Colonoscopy Image", type=["jpg", "png", "jpeg", "bmp"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    # Perform analysis
    with st.spinner("🔍 Analyzing image..."):
        # Polyp detection
        polyp_detected, confidence, yolo_results = detect_polyp(yolo_model, image)

        # Risk classification
        if polyp_detected:
            risk_level, risk_score, polyp_type = classify_risk(yolo_model, efficientnet_model, classifier_model, image)
        else:
            risk_level, risk_score, polyp_type = "N/A", 0.0, "None"
    
    # Display results
    with col1:
        st.subheader("📸 Analysis Image")
        if polyp_detected and yolo_results is not None:
            # Draw and show image with bounding boxes
            marked_image = draw_bounding_boxes(image, yolo_results)
            st.image(marked_image, caption="Detected Polyp (Boxed in Red)", use_column_width=True)
        else:
            st.image(image, caption="Uploaded Colonoscopy Image", use_column_width=True)
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        if polyp_detected:
            # Polyp Detection Results
            st.success("✅ **POLYP DETECTED**")
            
            # Confidence Meter
            st.metric("Detection Confidence", f"{confidence:.2%}")
            
            # Visual confidence bar
            st.progress(float(confidence), text=f"Confidence: {confidence:.1%}")
            
            st.divider()
            
            # Risk Assessment
            st.subheader("🏥 Risk Assessment")
            
            # Risk Level Color Coding
            if "High" in risk_level:
                st.error(f"⚠️ **{risk_level}**")
                risk_color = "🔴"
            elif "Medium" in risk_level:
                st.warning(f"⚠️ **{risk_level}**")
                risk_color = "🟡"
            else:
                st.info(f"ℹ️ **{risk_level}**")
                risk_color = "🟢"
            
            # Risk Score Display
            col_risk1, col_risk2 = st.columns(2)
            with col_risk1:
                st.metric("Risk Level", risk_level)
            with col_risk2:
                st.metric("Risk Score", f"{risk_score:.2%}")
            
            # Risk Score Visualization
            st.progress(float(risk_score), text=f"Risk Score: {risk_score:.1%}")
            
            # Detailed Assessment
            st.divider()
            st.subheader("📋 Assessment Summary")

            # Add polyp type display
            type_color = "🔴" if polyp_type == "Adenomatous" else ("🟢" if polyp_type == "Hyperplastic" else ("🟡" if polyp_type == "Serrated_Lesions" else "⚪"))

            # Display as individual metrics for better readability
            col_sum1, col_sum2 = st.columns(2)
            with col_sum1:
                st.metric("Polyp Detection", "✅ Positive")
                st.metric("Polyp Type", f"{type_color} {polyp_type}")
                st.metric("Risk Score", f"{risk_score:.1%}")
            with col_sum2:
                st.metric("Confidence Level", f"{confidence:.1%}")
                st.metric("Risk Classification", f"{risk_color} {risk_level}")
            
            st.warning("⚠️ This polyp requires medical evaluation.")
        else:
            st.success("✅ **NO POLYP DETECTED**")
            st.info("The colonoscopy image appears normal based on AI analysis.")
            st.divider()
            st.subheader("📋 Assessment Summary")
            
            # Display as individual metrics
            col_sum1, col_sum2 = st.columns(2)
            with col_sum1:
                st.metric("Polyp Detection", "✅ Negative")
                st.metric("Status", "Normal scan")
            with col_sum2:
                st.metric("Result", "No abnormal growths")
            
            st.success("✅ Continue regular screening as recommended by your healthcare provider.")
    
    # Generate and display report
    st.divider()
    st.subheader("📋 Medical Report")
    
    with st.spinner("📝 Generating patient-friendly report with Google Gemini..."):
        report = generate_report_with_gemini(api_key, polyp_detected, confidence, risk_level, risk_score, polyp_type)
    
    if report:
        st.markdown(report)
        
        # Download report
        report_bytes = report.encode('utf-8')
        st.download_button(
            label="📥 Download Report as Text",
            data=report_bytes,
            file_name="colonoscopy_report.txt",
            mime="text/plain"
        )
    else:
        st.error("❌ Failed to generate report. Please check your API key and try again.")

# Footer
st.divider()
st.markdown("""
---
**MEDICAL DISCLAIMER**: This application provides AI-assisted analysis for educational and preliminary screening purposes only. 
It is NOT a substitute for professional medical diagnosis or treatment. 
Always consult with a qualified healthcare professional for proper diagnosis and treatment.
""")