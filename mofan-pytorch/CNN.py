import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ===== 1. 数据 =====
transform = transforms.ToTensor()
train_ds = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)


# ===== 2. 模型 =====
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)   # -> [8, 26, 26]
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)  # -> [16, 24, 24]
        self.pool = nn.MaxPool2d(2, 2)                # -> [16, 12, 12]
        self.fc = nn.Linear(16 * 12 * 12, 10)

    def forward(self, x):
        print("input:", x.shape)  # [64, 1, 28, 28]

        x = self.conv1(x)
        print("after conv1:", x.shape)

        x = self.conv2(x)
        print("after conv2:", x.shape)

        x = self.pool(x)
        print("after pool:", x.shape)

        x = torch.flatten(x, 1)
        print("after flatten:", x.shape)

        out = self.fc(x)
        print("after fc:", out.shape)
        return out


model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()



# ===== 3. 训练一步（只跑一批就够）=====
images, labels = next(iter(train_loader))
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()

print("Loss:", loss.item())