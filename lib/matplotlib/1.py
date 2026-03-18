import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Параметры
device = torch.device("cpu")
LATENT_DIM = 100
BATCH_SIZE = 64
EPOCHS = 100

# 1) Данные: сразу приводим к диапазону [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))   # [0,1] -> [-1,1]
])

dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Чтобы было похоже на твой вариант: берём 15000 изображений
dataset = Subset(dataset, range(15000))
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 2) Генератор: шум -> изображение 28x28
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512, momentum=0.8),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256, momentum=0.8),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128, momentum=0.8),

            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.model(z)
        return x.view(-1, 1, 28, 28)

# 3) Дискриминатор: изображение -> вероятность, что оно настоящее
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Один и тот же шум для проверки прогресса
fixed_noise = torch.randn(4, LATENT_DIM, device=device)

def show_images(epoch):
    G.eval()
    with torch.no_grad():
        imgs = G(fixed_noise).cpu()
    plt.figure(figsize=(4, 4))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow((imgs[i, 0] + 1) / 2, cmap="gray")  # обратно в [0,1]
        plt.axis("off")
    plt.suptitle(f"Epoch {epoch}")
    plt.tight_layout()
    plt.show()
    G.train()

# 4) Обучение
for epoch in range(EPOCHS):
    for real_imgs, _ in loader:
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        # ---- Обучаем дискриминатор ----
        optimizer_D.zero_grad()

        z = torch.randn(batch_size, LATENT_DIM, device=device)
        fake_imgs = G(z)

        real_loss = criterion(D(real_imgs), valid)
        fake_loss = criterion(D(fake_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # ---- Обучаем генератор ----
        optimizer_G.zero_grad()

        g_loss = criterion(D(fake_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | D loss = {d_loss.item():.4f} | G loss = {g_loss.item():.4f}")

    if (epoch + 1) % 10 == 0:
        show_images(epoch + 1)