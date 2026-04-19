import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import PrunableNet
from utils import sparsity_loss, calculate_sparsity, test
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

model = PrunableNet().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
lambda_val = 0.01

for epoch in range(5):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss_cls = criterion(outputs, labels)
        loss_sparse = sparsity_loss(model)

        loss = loss_cls + lambda_val * loss_sparse

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

accuracy = test(model, test_loader)
sparsity = calculate_sparsity(model)

print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Sparsity Level: {sparsity:.2f}%")

gates_all = []

for module in model.modules():
    if hasattr(module, "gate_scores"):
        gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy()
        gates_all.extend(gates.flatten())

plt.hist(gates_all, bins=50)
plt.title("Gate Distribution")
plt.show()