import os
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
import imageio

# ========= CONFIGURAÇÕES =========
IMG_SIZE = 28
BATCH_SIZE = 128
DATA_DIR = './data'
IMG_DIR = './results'

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# ========= TRANSFORMAÇÃO E DATASET =========
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

fashion_data = datasets.FashionMNIST(
    DATA_DIR,
    transform=transform,
    download=True
)

dataloader = DataLoader(fashion_data, batch_size=BATCH_SIZE, shuffle=True)

# ========= VISUALIZAÇÃO DO DATASET =========
def lookat_dataset(dataset):
    figure = plt.figure(figsize=(16, 4))
    rows, cols = 2, 8
    for i in range(1, 17):
        idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[idx]
        figure.add_subplot(rows, cols, i)
        plt.axis("off")
        img = (img.squeeze() * 0.5) + 0.5
        plt.imshow(img, cmap='gray')
    plt.show()

# ========= MODELOS =========
class Generator(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_in, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self, z):
        return self.layers(z)

class Discriminator(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        n_out = 1
        self.layers = nn.Sequential(
            nn.Linear(n_in, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, n_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

# ========= DISPOSITIVO =========
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Rodando na {device}")

# ========= INSTÂNCIAS =========
noise_dim = 100
generator = Generator(noise_dim, IMG_SIZE * IMG_SIZE).to(device)
discriminator = Discriminator(IMG_SIZE * IMG_SIZE).to(device)

# ========= FUNÇÕES AUXILIARES =========
def images_to_vectors(images):
    return images.view(images.size(0), IMG_SIZE * IMG_SIZE)

def vectors_to_images(vectors, nc=1):
    return vectors.view(vectors.size(0), nc, IMG_SIZE, IMG_SIZE)

def noise(size, dim=noise_dim):
    return torch.randn(size, dim).to(device)

def log_images(test_images, savepath=None):
    figure = plt.figure(figsize=(8, 8))
    figure.subplots_adjust(wspace=-0.08, hspace=0.01)
    rows, cols = len(test_images) // 4, 4
    for i, img in enumerate(test_images):
        figure.add_subplot(rows, cols, i + 1)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap='gray')
    if savepath:
        figure.savefig(savepath)
    plt.close(figure)

# ========= LABELS =========
def real_data_target(size):
    return torch.ones(size, 1).to(device)

def fake_data_target(size):
    return torch.zeros(size, 1).to(device)

# ========= TREINAMENTO =========
lr = 0.0002
loss_fn = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=lr)
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

def train_discriminator(optimizer, real_data, fake_data):
    optimizer.zero_grad()
    pred_real = discriminator(real_data)
    error_real = loss_fn(pred_real, real_data_target(real_data.size(0)))
    error_real.backward()

    pred_fake = discriminator(fake_data)
    error_fake = loss_fn(pred_fake, fake_data_target(fake_data.size(0)))
    error_fake.backward()

    optimizer.step()
    return error_real + error_fake, pred_real, pred_fake

def train_generator(optimizer, fake_data):
    optimizer.zero_grad()
    pred = discriminator(fake_data)
    error = loss_fn(pred, real_data_target(pred.size(0)))
    error.backward()
    optimizer.step()
    return error

# ========= LOOP DE TREINO =========
num_epochs = 20
G_losses = []
D_losses = []

num_test_samples = 16
torch.manual_seed(7777)
test_noise = noise(num_test_samples)
imagepath = os.path.join(IMG_DIR, f'epoch_0.jpg')

# Primeira geração (antes de treinar)
log_images(vectors_to_images(generator(test_noise)).cpu().detach().numpy(), imagepath)

for epoch in range(num_epochs):
    for realbatch, _ in dataloader:
        real_data = images_to_vectors(realbatch).to(device)
        fake_data = generator(noise(real_data.size(0)))

        d_error, dpred_real, dpred_fake = train_discriminator(d_optimizer, real_data, fake_data)

        fake_data = generator(noise(realbatch.size(0)))
        g_error = train_generator(g_optimizer, fake_data)

    # Salva amostra do progresso
    imagepath = os.path.join(IMG_DIR, f'epoch_{epoch+1}.jpg')
    test_images = vectors_to_images(generator(test_noise)).cpu().detach().numpy()
    log_images(test_images, imagepath)

    G_losses.append(g_error)
    D_losses.append(d_error)

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Loss D: {d_error:.4f} | Loss G: {g_error:.4f}")
    print(f"D(x): {dpred_real.mean():.4f} | D(G(z)): {dpred_fake.mean():.4f}")
    print("-" * 40)

# ========= PLOT DE PERDAS =========
def plot_losses(losses):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.gca()
    for name, values in losses.items():
        values = [v.item() for v in values]
        ax.plot(values, label=name)
    ax.legend(fontsize="14")
    ax.set_xlabel("Epoch", fontsize="14")
    ax.set_ylabel("Loss", fontsize="14")
    ax.set_title("Loss vs Epochs", fontsize="16")
    plt.show()

plot_losses({"Generator": G_losses, "Discriminator": D_losses})

# ========= GERA GIF =========
images = []
for filename in sorted(os.listdir(IMG_DIR), key=lambda x: int(x.split('_')[1].split('.')[0])):
    if filename.endswith('.jpg'):
        filepath = os.path.join(IMG_DIR, filename)
        images.append(imageio.imread(filepath))

imageio.mimsave('fashion_training.gif', images)
print("GIF gerado: fashion_training.gif ✅")
