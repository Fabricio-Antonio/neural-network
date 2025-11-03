import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ======== CONFIGURAÇÃO DO DISPOSITIVO ===========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== TAMANHO DAS IMAGENS ===========
IMAGE_SIZE = (10, 6)

# ======== DATASET ===========
class AlgebraicDataset(Dataset):
    def __init__(self, func, interval, nsamples):
        self.x = np.linspace(interval[0], interval[1], nsamples)
        self.y = func(self.x)
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ======== REDE NEURAL ===========
class MultiLayerNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.layers(x)

# ======== FUNÇÕES AUXILIARES ===========
def train(model, dataloader, lossfunc, optimizer):
    model.train()
    cumloss = 0.0
    for x, y in dataloader:
        x = x.unsqueeze(1).to(device)
        y = y.unsqueeze(1).to(device)
        pred = model(x)
        loss = lossfunc(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cumloss += loss.item()
    return cumloss / len(dataloader)

def test(model, dataloader, lossfunc):
    model.eval()
    tot_loss = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.unsqueeze(1).to(device)
            y = y.unsqueeze(1).to(device)
            pred = model(x)
            loss = lossfunc(pred, y)
            tot_loss += loss.item()
    return tot_loss / len(dataloader)

def generate_plot(f, model, nsamples=100, interval=(-10, 10)):
    """Gera e retorna uma figura matplotlib sem exibir."""
    model.eval()
    x = np.linspace(interval[0], interval[1], nsamples)
    y_true = f(x)
    with torch.no_grad():
        y_pred = model(torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(device)).cpu().numpy()

    fig, ax = plt.subplots(figsize=IMAGE_SIZE)
    ax.plot(x, y_true, label="Função Real", linewidth=2)
    ax.plot(x, y_pred, label="Função Predita", linestyle="--", linewidth=2)
    ax.legend(fontsize=12)
    ax.grid(True)
    ax.set_title("Comparação Real x Predito", fontsize=14)
    return fig

# ======== CONFIGURAÇÕES DE TREINO ===========
f = lambda x: np.cos(x / 2)
interval = (-10, 10)
train_nsamples = 200
test_nsamples = 100

train_dataset = AlgebraicDataset(f, interval, train_nsamples)
test_dataset = AlgebraicDataset(f, interval, test_nsamples)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = MultiLayerNetwork().to(device)
lossfunc = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# ======== LOOP DE TREINAMENTO ===========
epochs = 50001
saved_figures = []

for t in range(epochs):
    train_loss = train(model, train_dataloader, lossfunc, optimizer)
    if t % 500 == 0:
        print(f"Epoch: {t} | Train Loss: {train_loss:.6f}")
        fig = generate_plot(f, model, nsamples=200, interval=interval)
        saved_figures.append((t, fig))

# ======== AVALIAÇÃO FINAL ===========
test_loss = test(model, test_dataloader, lossfunc)
print(f"Test Loss: {test_loss:.6f}")

# ======== SALVAR AS FIGURAS EM PDF (NA PASTA DOWNLOADS) ===========
downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
os.makedirs(downloads_dir, exist_ok=True)

pdf_filename = os.path.join(downloads_dir, "treinamento_resultados.pdf")

with PdfPages(pdf_filename) as pdf:
    for epoch, fig in saved_figures:
        fig.suptitle(f"Epoch {epoch}", fontsize=16, weight='bold')
        pdf.savefig(fig)
        plt.close(fig)

print(f"\n✅ PDF gerado com sucesso em: {pdf_filename}")
