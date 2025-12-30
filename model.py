import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.utils.hipify.hipify_python import preprocessor
from torchvision import models


def get_resnet_model(num_classes=2):
    # ResNet18 hafif ve hızlıdır, X-ray için idealdir
    model = models.resnet18(weights='IMAGENET1K_V1')

    # Son katmanı (fully connected) değiştiriyoruz
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

#1- Ayarlamalar
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
class_names = ["NORMAL", "PNEUMONIA"]

#Modeli Yukleme
model = get_resnet_model()
model.to(device)
model.eval()

#2.OpevCV ile Görüntü işleme fonksiyonu

def predict_and_show(image_path):
    #OpenCV ile okuma
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    #Modelin Bekledği Preproccesing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #PIL image çevirip preprocces uygulanması
    input_tensor = preprocess(Image.fromarray(img_rgb)).unsqueeze(0).to(device)

    #prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)
        prob = torch.nn.functional.softmax(outputs, dim=1)[0]

    label = class_names[preds[0]]
    confidence = prob[preds[0]].item() * 100

    #OpenCV ile resmi ekrana yazdırma
    color = (0, 255, 0) if label == 'NORMAL' else (0,0,255) #Sağlıklı yeşil - Hasta kırmızı renkte

    #Görüntüye yazı ekleme
    text = f"{label}: {confidence:.2f}%"
    cv2.putText(img_bgr, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    #Pencereyi gosterme
    cv2.imshow('X-ray Diognosis', img_bgr)
    cv2.waitKey(0),
    cv2.destroyAllWindows()