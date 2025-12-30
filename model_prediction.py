import torch

from model import get_resnet_model, predict_and_show

# main.py içindeki model yükleme kısmını şu hale getir:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_resnet_model(num_classes=2)

# EĞİTİLMİŞ AĞIRLIKLARI BURADA YÜKLÜYORUZ
model.load_state_dict(torch.load('chest_resnet18.pth', map_location=device))
model.to(device)
model.eval() # Test moduna almayı unutma!

import os
import random

test_path = "data/chest_xray/test/PNEUMONIA"
random_file = random.choice(os.listdir(test_path))
full_path = os.path.join(test_path, random_file)

# Fonksiyonu çağır
predict_and_show(full_path)