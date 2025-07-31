import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ----------------------- Define the model class --------------------------
class FaceMaskClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# ----------------------- Image Transform --------------------------
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------- Load the model --------------------------
num_classes = 2
model = FaceMaskClassifierCNN(num_classes=num_classes)
model.load_state_dict(torch.load("face_detection_model.pth", map_location=torch.device('cpu')))
model.eval()

class_labels = ['With Mask', 'Without Mask']

# ----------------------- Streamlit UI --------------------------
st.title("😷 Face Mask Detector")
st.markdown("Upload a face image or use your webcam to check if you're wearing a mask.")

# Input options
st.subheader("Step 1: Select Image Source")
option = st.radio("Choose input method:", ("Upload from device", "Use webcam"))

image = None

if option == "Upload from device":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

elif option == "Use webcam":
    camera_image = st.camera_input("Take a photo using webcam")
    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

# ----------------------- Prediction --------------------------
if image is not None:
    # Prediction First
    input_tensor = image_transforms(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = class_labels[predicted.item()]
        conf_percent = confidence.item() * 100

    # Show Prediction
    if conf_percent < 80:
        st.warning(f"⚠️ Low Confidence: {conf_percent:.2f}%")
        st.info("Please share a photo with a **clear face close to the camera**.")
    else:
        if predicted_class == 'With Mask':
            st.success(f"✅ Prediction: **{predicted_class}** ({conf_percent:.2f}% confidence)")
        else:
            st.error(f"🚫 Prediction: **{predicted_class}** ({conf_percent:.2f}% confidence)")

    # Then Show Image
    st.image(image, caption="Input Image", use_container_width=True)
