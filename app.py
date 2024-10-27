import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

# Define the number of classes
n_classes = 5

# Load the pre-trained ResNet-50 model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, n_classes)

# Load the state dictionary
state_dict = torch.load('entire_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Define class labels
classes = [
    'Light Diseases and Disorders of Pigmentation',
    'Acne and Rosacea Photos',
    'Poison Ivy Photos and other Contact Dermatitis',     
    'Atopic Dermatitis Photos',
    'Hair Loss Photos Alopecia and other Hair Diseases',
]

# Debugging transformation (No Resize/Normalization)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to match model input
    transforms.ToTensor() 
])


# Streamlit app
st.title("Skin Disease Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    input_tensor = transform(image).unsqueeze(0)

    # Make predictions
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)

    # Debug: Print output logits and predicted index
    st.write(f"Model output logits: {output}")
    st.write(f"Predicted index: {predicted_idx.item()}")

    predicted_class = classes[predicted_idx.item()]
    st.write(f"Prediction: {predicted_class}")
