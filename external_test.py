
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =====================
# PATHS (Drive yapına göre)
# =====================
MODEL_PATH = "/content/drive/MyDrive/GLOKOM PROJE/densenet121_best.pt"
TEST_DIR = "/content/drive/MyDrive/GLOKOM PROJE/New_Test_Data"

CLASSES = ["normal", "glokom"]

# =====================
# TRANSFORMS
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================
# LOAD MODEL
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet121(weights=None)
model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# =====================
# LOAD TEST DATA
# =====================
y_true = []
y_pred = []

for label, class_name in enumerate(CLASSES):
    class_dir = os.path.join(TEST_DIR, class_name)
    for file in os.listdir(class_dir):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(class_dir, file)
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image)
                pred = torch.argmax(outputs, dim=1).item()

            y_true.append(label)
            y_pred.append(pred)

# =====================
# RESULTS
# =====================
print("\n📊 EXTERNAL TEST SET RESULTS\n")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))
