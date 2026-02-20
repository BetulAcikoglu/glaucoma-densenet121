import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse

def load_model(model_path, device):
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(image_path, model, device):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred = torch.argmax(probs).item()

    classes = ['normal', 'glokom']
    return classes[pred], probs[pred].item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to fundus image')
    parser.add_argument('--model', type=str, default='densenet121_best.pt', help='Model path')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.model, device)
    label, conf = predict(args.image, model, device)

    print(f'Prediction: {label}')
    print(f'Confidence: {conf:.4f}')
