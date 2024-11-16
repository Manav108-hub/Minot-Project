import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import time

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(512, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        
        x = self.global_pool(x)
        x = x.view(-1, 512)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

@st.cache_resource
def load_model(model_path, num_classes, device):
    model = CNNModel(num_classes=num_classes)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_species(image, model, transform, classes, device, threshold=0.7):
    try:
        start_time = time.time()
        
        # Preprocess image
        image_tensor = transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            # Get top 3 predictions
            top_probs, top_classes = torch.topk(probabilities, 3)
            predictions = [(classes[idx], prob.item()) 
                         for idx, prob in zip(top_classes[0], top_probs[0])]
            
        processing_time = time.time() - start_time
        
        if confidence.item() < threshold:
            return None, None, predictions, processing_time
        
        return classes[predicted_class.item()], confidence.item(), predictions, processing_time
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None, None, None

def main():
    # Page configuration
    st.set_page_config(
        page_title="Species Recognition",
        page_icon="🦁",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .prediction-box {
            padding: 20px;
            border-radius: 10px;
            background-color: #f0f2f6;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .confidence-high { 
            color: #0f5132;
            font-weight: bold;
        }
        .confidence-medium { 
            color: #664d03;
            font-weight: bold;
        }
        .confidence-low { 
            color: #842029;
            font-weight: bold;
        }
        .stApp {
            background-color: #f8f9fa;
        }
        .main-header {
            text-align: center;
            padding: 1rem;
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<div class="main-header"><h1>🦁 Species Recognition</h1></div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write("""
        This application uses deep learning to recognize different animal species.
        Simply upload an image, and the model will analyze it to identify the species.
        """)
        st.markdown("---")
        st.write("Model Information:")
        st.write("- Architecture: CNN")
        st.write("- Input Size: 224x224")
        st.write("- Confidence Threshold: 70%")
    
    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "E:/Minor-Project/Datasets/Train/animals"
    model_path = "E:/Minor-Project/best_model.pth"
    
    # Input preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    classes = sorted(os.listdir(data_dir))
    num_classes = len(classes)
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model(model_path, num_classes, device)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload an image of an animal to identify its species"
    )
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Uploaded Image")
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("🔍 Recognition Results")
            with st.spinner("Analyzing image..."):
                species, confidence, top_predictions, proc_time = predict_species(
                    image, model, transform, classes, device
                )
                
                if top_predictions:
                    st.success(f"✨ Analysis Complete! ({proc_time:.2f} seconds)")
                    
                    # Display top predictions
                    for i, (pred_species, pred_conf) in enumerate(top_predictions, 1):
                        confidence_class = (
                            "confidence-high" if pred_conf > 0.7
                            else "confidence-medium" if pred_conf > 0.4
                            else "confidence-low"
                        )
                        
                        st.markdown(f"""
                            <div class="prediction-box">
                                <h3>#{i} Prediction</h3>
                                <span class="{confidence_class}">
                                    {pred_species}: {pred_conf*100:.1f}%
                                </span>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        if pred_conf > 0.4:
                            st.progress(pred_conf)

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 Features")
        st.write("- Real-time analysis")
        st.write("- Top 3 predictions")
        st.write("- Confidence scores")
    
    with col2:
        st.markdown("### 🎯 Accuracy")
        st.write("- High precision model")
        st.write("- Confidence threshold")
        st.write("- Processing time tracking")
    
    with col3:
        st.markdown("### 💡 Tips")
        st.write("- Use clear images")
        st.write("- Center the subject")
        st.write("- Good lighting helps")

    st.markdown("<div style='text-align: center; color: gray;'>Powered by Deep Learning | Created with ❤️</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()