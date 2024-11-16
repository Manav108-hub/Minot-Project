import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from pathlib import Path

# Define the CNN architecture (same as your original model)
class CNNModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)

        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)

        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(256)

        self.fc1 = torch.nn.Linear(256 * 14 * 14, 512)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))

        x = x.view(-1, 256 * 14 * 14)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Function to compute mean and std (from your prediction.py)
def compute_mean_std(data_dir):
    means = []
    stds = []
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = Image.open(img_path).convert("RGB")
                img = np.array(img)
                means.append(np.mean(img, axis=(0, 1)))
                stds.append(np.std(img, axis=(0, 1)))

    means = np.mean(means, axis=0)
    stds = np.mean(stds, axis=0)
    return means, stds

# Function to load the model
@st.cache_resource
def load_model(model_path, num_classes, device):
    model = CNNModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

# Function to predict species
def predict_species(image, model, transform, classes, device):
    try:
        image = transform(image).unsqueeze(0)
        image = image.to(device)
        
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
        return classes[predicted_class.item()], confidence.item()
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None

def main():
    st.set_page_config(
        page_title="Automated Species Recognition",
        page_icon="🦁",
        layout="wide"
    )

    # Add CSS styling
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #388E3C;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("🦁 Automated Species Recognition")
    st.markdown("Discover the wonders of wildlife with AI-powered recognition.")

    # About section
    with st.expander("ℹ️ About"):
        st.markdown("""
        This application uses advanced AI technology to identify animal species from images. 
        Our system employs Convolutional Neural Networks (CNNs) to provide accurate species recognition, 
        helping researchers and wildlife enthusiasts in their work.
        """)

    # Setup model and transformations
    data_dir = "E:/Minor-Project/Datasets/Train/animals"  # Update this path
    model_path = "E:/Minor-Project/animal_species_cnn.pth"  # Update this path
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(os.listdir(data_dir))
    classes = sorted(os.listdir(data_dir))
    
    mean, std = compute_mean_std(data_dir)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Load model
    model = load_model(model_path, num_classes, device)

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display image and make prediction
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            st.subheader("Recognition Results")
            with st.spinner("Analyzing image..."):
                species, confidence = predict_species(image, model, transform, classes, device)
                
                if species and confidence:
                    st.success("Analysis Complete!")
                    st.markdown(f"**Species**: {species}")
                    st.markdown(f"**Confidence**: {confidence*100:.2f}%")
                    
                    # Display confidence bar
                    st.progress(confidence)

    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ by Your Name")

if __name__ == "__main__":
    main()