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
    classes[0]: [' Vitamin D synthesis is triggered by sunlight, and adequate levels support overall skin health. While a deficiency doesn’t cause light diseases directly, proper vitamin D levels are crucial for maintaining skin integrity and managing light sensitivity.  Vitamin B12: Vitamin B12 deficiency can lead to hyperpigmentation and other skin changes. Proper levels of vitamin B12 are important for overall skin health and preventing pigmentary changes.'],

    classes[1]: ['Vitamin D**: A deficiency in vitamin D can lead to inflammation, which may exacerbate acne. Vitamin D has anti-inflammatory properties, and low levels are associated with increased skin inflammation.Vitamin A**: Vitamin A plays a critical role in skin health. A deficiency may lead to excessive sebum production, clogged pores, and the formation of acne.Zinc (a mineral)**: Zinc deficiency is also linked to acne. Zinc helps with wound healing, reduces inflammation, and limits the growth of acne-causing bacteria.'],
    classes[2]: ['**Vitamin D**: Poison ivy reactions involve immune responses that can be affected by vitamin D levels. Vitamin D modulates the immune system, helping reduce inflammation caused by contact dermatitis. **Vitamin C**: Vitamin C supports collagen synthesis and accelerates wound healing, aiding in recovery from skin reactions like poison ivy. **Vitamin E**: Vitamin E has antioxidant properties that help protect the skin and may reduce damage caused by exposure to allergens like poison ivy. **Zinc**: Zinc can reduce inflammation and promote healing of skin rashes caused by contact dermatitis. Zinc creams or supplements may alleviate symptoms.'],
    classes[3]: ['1. **Vitamin B2 (Riboflavin) - Deficiency in riboflavin can lead to seborrheic dermatitis, particularly around the nose, eyes, and ears. Riboflavin is essential for maintaining healthy skin, and a lack of it may cause redness, inflammation, and flaky skin. **Vitamin B3 (Niacin)**: - Niacin deficiency can cause a condition known as **pellagra**, which includes dermatitis as one of its primary symptoms. Pellagra results in inflamed, scaly, and thickened skin, especially in areas exposed to sunlight. **Vitamin B7 (Biotin)**:Biotin deficiency is linked to seborrheic dermatitis. Biotin is important for skin health, and low levels may lead to dry, flaky skin.**Vitamin C**: A deficiency in vitamin C can result in weakened skin structure due to poor collagen production, which may contribute to dermatitis and other skin problems.*Vitamin D**: Low levels of vitamin D have been associated with an increased risk of atopic dermatitis (eczema). Vitamin D helps regulate the immune system and inflammatory responses in the skin.'],
    classes[4]: ['**Vitamin B12**:**Role**: Vitamin B12 is important for red blood cell production and maintaining healthy hair follicles. Deficiency can lead to hair loss due to impaired cell metabolism and reduced oxygen delivery to hair follicles.**Source**: Katta, R., & Desai, S. (2013). Vitamin Deficiencies and Hair Loss: A Review. *Dermatologic Therapy*, 26(5), 322-330.**Biotin (Vitamin B7)**: **Role**: Biotin is often associated with healthy hair growth. A deficiency can cause hair thinning and loss, as it plays a role in the synthesis of keratin, a protein important for hair structure.**Source**: Zempleni, J., & Rucker, R. B. (2005). Biotin. In: *Present Knowledge in Nutrition*, 10th edition, pp. 354-364.**Vitamin A**:**Role**: Vitamin A supports the health of skin cells, including those in the scalp. Both deficiency and excess can cause hair loss. Deficiency leads to dry and brittle hair, while excess can lead to toxicity and hair loss.**Vitamin E**:**Role**: Vitamin E has antioxidant properties that help protect hair follicles from oxidative stress. A deficiency may contribute to hair loss, although its role is less direct compared to other vitamins.'],
}

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to match model input
    transforms.ToTensor(),         # Convert to tensor
])

# Streamlit app
st.title("Skin Disease Classifier")

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

        # Debug: Print output logits and predicted index
        # st.write(f"Model output logits: {output}")
        # st.write(f"Predicted index: {predicted_idx.item()}")

        # Map prediction to class
        predicted_class = classes[predicted_idx.item()]
        st.write(f"Prediction: {predicted_class}")
        st.write(f"Vitamin Info: {temp[predicted_class]}")
