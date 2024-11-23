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

# Define additional information for predictions
temp = {
    classes[0]: ['Vitamin D', 'Vitamin B12'],
    classes[1]: ['Vitamin D', 'Vitamin A', 'Zinc'],
    classes[2]: ['Vitamin D', 'Vitamin C', 'Vitamin E', 'Zinc'],
    classes[3]: ['Vitamin B2', 'Vitamin B3', 'Vitamin B7', 'Vitamin C', 'Vitamin D'],
    classes[4]: ['Vitamin B12', 'Biotin', 'Vitamin A', 'Vitamin E'],
}

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to match model input
    transforms.ToTensor(),         # Convert to tensor
])

# Streamlit app
st.title("Vitamin Deficiency Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file).convert("RGB")  # Ensure image has 3 channels
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension, shape: [1, 3, 256, 256]

    # Check input tensor shape
    if input_tensor.shape[1:] != (3, 256, 256):
        st.error(f"Expected input shape [1, 3, 256, 256], but got {input_tensor.shape}")
    else:
        # Make predictions
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_idx = torch.max(output, 1)

        # Map prediction to class
        predicted_class = classes[predicted_idx.item()]
        
        # Retrieve relevant deficiencies (up to 2) or show "No deficiency"
        deficiencies = temp.get(predicted_class, [])
        if deficiencies:
            st.write(f"Prediction: {predicted_class}")
            st.write("Deficiencies Detected: " + ", ".join(deficiencies[:2]))
        else:
            st.write("Prediction: No deficiency detected")
