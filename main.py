import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.init import trunc_normal_
from torchvision import datasets, transforms, models
import os

from model import get_resnet_model

data_dir = "data/chest_xray"
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val', 'test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True)
              for x in ['train', 'val', 'test']}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_resnet_model(num_classes=2)  # Senin fonksiyonun
model.to(device)

# 4. Kayıp ve Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 5. EĞİTİM DÖNGÜSÜ
print("Eğitim başlıyor... Arkana yaslan.")
for epoch in range(5):  # 5 tur dönecek
    model.train()
    running_loss = 0.0

    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1} bitti. Kayıp (Loss): {running_loss / len(dataloaders['train']):.4f}")

# 6. Kaydet (OpenCV kodunda bunu çağıracağız)
torch.save(model.state_dict(), "chest_resnet18.pth")
print("Model 'chest_resnet18.pth' adıyla kaydedildi!")