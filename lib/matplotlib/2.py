from PIL import Image
import imageio
import cv2
import numpy as np
import os
from torchvision import datasets, transforms
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time

warnings.filterwarnings('ignore')

# Убираем вывод информации о CUDA, так как теперь используем CPU
print("Используется CPU для вычислений")

data_directory = './my_fashion_mnist_data'
transform = transforms.ToTensor()

# 1. Загружаем датасет (пиксели в [0,1])
train_dataset = datasets.FashionMNIST(
    root=data_directory,
    train=True,
    download=True,
    transform=transform
)
num_samples = 15000
images_flat = []

# 2. Извлекаем изображения, преобразуем в вектор и сохраняем
for i in tqdm(range(num_samples), desc="Обработка изображений"):
    image, _ = train_dataset[i]  # image - тензор [1,28,28], значения в [0,1]
    image_np = image.squeeze().numpy().flatten()  # (784,)
    images_flat.append(image_np)

# 3. Сохраняем в CSV
images_flat = np.array(images_flat, dtype=np.float32)
columns = [f"pixel_{i}" for i in range(784)]
df = pd.DataFrame(images_flat, columns=columns)
csv_file_path = "fashion_mnist_train.csv"
df.to_csv(csv_file_path, index=False)
print(f"Сохранено: {csv_file_path}")

# 4. Загрузка из CSV
dataMF = pd.read_csv(csv_file_path)
X_train = dataMF.values.astype(np.float32)  # (15000, 784)

# 5. Преобразуем в формат (N, 1, 28, 28) для PyTorch
X_train = X_train.reshape(-1, 28, 28, 1)
X_train = np.transpose(X_train, (0, 3, 1, 2))  # (N, C=1, H=28, W=28)

print("Форма после reshape и transpose:", X_train.shape)  # (15000,1,28,28)
print("Мин и макс до нормализации:", X_train.min(), X_train.max())  # должно быть 0.0 и 1.0

dataMF.head()

# 6. Нормализация из [0,1] в [-1,1]
X_train = X_train * 2.0 - 1.0
print("Мин и макс после нормализации:", X_train.min(), X_train.max())  # около -1.0 и 1.0

# 7. Преобразуем в torch.Tensor для обучения (устройство по умолчанию - CPU)
X_train = torch.tensor(X_train)

# Визуализация 25 изображений
fig, axes = plt.subplots(5, 5, figsize=(8,8))
idx = 0
for i in range(5):
    for j in range(5):
        axes[i, j].imshow(X_train[idx, 0], cmap='gray')
        axes[i, j].axis('off')
        idx += 1
plt.show()

# Определение генератора GAN
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512, momentum=0.8),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256, momentum=0.8),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128, momentum=0.8),
            
            nn.Linear(128, 784),
            nn.Tanh()  # Генерируем пиксели в [-1,1]
        )
    def forward(self, z):
        out = self.model(z)
        out = out.view(-1, 1, 28, 28)  # Формат NCHW
        return out
    
generator = Generator()
print(generator)

# Определение дискриминатора GAN
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
    
discriminator = Discriminator()
print(discriminator)

# Создаём случайный шум (z) - в PyTorch: torch.randn
noise = torch.randn(1, 100)

# Генерируем изображение
generator.eval()  # отключаем dropout/batchnorm в eval режиме
with torch.no_grad():
    img = generator(noise).cpu().numpy()
plt.imshow(img[0, 0, :, :])
plt.axis('off')
plt.show()

# Объявляем оптимизаторы и функцию потерь
adversarial_loss = nn.BCELoss()  # бинарная кросс-энтропия

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

BATCH_SIZE = 64
EPOCHS = 150
# Принудительно используем CPU
device = torch.device('cpu')

# Переносим модели на устройство (теперь CPU)
generator.to(device)
discriminator.to(device)

# Создаем DataLoader из X_train (тензор)
dataset = TensorDataset(X_train)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Функция для визуализации сгенерированных изображений
def plot_generated_images(generator, epoch, samples=4):
    generator.eval()
    noise = torch.randn(samples, 100).to(device)
    with torch.no_grad():
        generated_images = generator(noise).cpu().numpy()
    plt.figure(figsize=(4, 4))
    for k in range(samples):
        plt.subplot(2, 2, k + 1)
        plt.imshow(generated_images[k, 0])
        plt.xticks([])
        plt.yticks([])
    plt.suptitle(f'Эпоха {epoch+1}')
    plt.tight_layout()
    plt.show()
    generator.train()
    
# Основной цикл обучения
start_time = time.time()  # Засекаем общее время обучения
for epoch in range(EPOCHS):
    epoch_start = time.time()  # Время начала эпохи
    print(f"Эпоха {epoch + 1}/{EPOCHS}")
    for i, (real_images,) in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        # Метки для реальных и фейковых изображений
        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)
        
        # --------------------
        # 1. Обучение дискриминатора
        optimizer_D.zero_grad()
        # Реальные изображения
        pred_real = discriminator(real_images)
        loss_real = adversarial_loss(pred_real, valid)
        # Фейковые изображения
        noise = torch.randn(batch_size, 100).to(device)
        gen_images = generator(noise)
        pred_fake = discriminator(gen_images.detach())
        loss_fake = adversarial_loss(pred_fake, fake)
        d_loss = (loss_real + loss_fake) / 2
        d_loss.backward()
        optimizer_D.step()
        
        # --------------------
        # 2. Обучение генератора
        optimizer_G.zero_grad()
        pred_gen = discriminator(gen_images)
        g_loss = adversarial_loss(pred_gen, valid)  # хотим, чтобы дискриминатор думал, что это настоящие
        g_loss.backward()
        optimizer_G.step()
        
        # Логирование каждые 100 батчей
        if (i + 1) % 100 == 0:
            elapsed_batch = time.time() - epoch_start
            print(f"  Батч {i+1}/{len(dataloader)} - D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}, "
                  f"Время с начала эпохи: {elapsed_batch:.1f} сек")
    
    # Визуализация каждые 10 эпох
    epoch_time = time.time() - epoch_start
    print(f"Эпоха {epoch + 1} завершена за {epoch_time:.1f} секунд")
    if (epoch + 1) % 10 == 0:
        plot_generated_images(generator, epoch)

total_time = time.time() - start_time
print(f"Общее время обучения: {total_time // 60:.0f} мин {total_time % 60:.1f} сек")
print("Обучение завершено")