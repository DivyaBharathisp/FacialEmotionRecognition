import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Import your model architecture
from Facial_Expression_Model import FullExpressionModel, BarlowTwinsLoss

# --------------------
# 1. Dataset & Loader
# --------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder('fer2013/train', transform=transform)
test_dataset = datasets.ImageFolder('fer2013/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --------------------
# 2. Model Setup
# --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(train_dataset.classes)

model = FullExpressionModel(num_classes=num_classes).to(device)
criterion_ssl = BarlowTwinsLoss()
criterion_cls = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --------------------
# 3. Training Loop
# --------------------
EPOCHS = 1

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for (img1, label), (img2, _) in zip(train_loader, train_loader):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        # SSL loss
        z1, z2 = model(img1, img2, mode="train")
        ssl_loss = criterion_ssl(z1, z2)

        # Classification loss
        logits = model(img1, mode="eval")
        cls_loss = criterion_cls(logits, label)

        loss = ssl_loss + cls_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Accuracy calculation
        _, predicted = torch.max(logits.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss:.4f}, Training Accuracy: {acc:.2f}%")

# --------------------
# 4. Evaluation
# --------------------
def evaluate(model, dataloader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, mode="eval")
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")

evaluate(model, test_loader)

# --------------------
# 5. Save Model
# --------------------
torch.save(model.state_dict(), "fer_cbam_barlow_model.pth")

