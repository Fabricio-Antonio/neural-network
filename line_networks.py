import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.distributions.uniform as urand
import matplotlib.pyplot as plt

# ======== MODELO ===========
class LineNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 1)
        )

    def forward(self, x):
        return self.layers(x)


# ======== DATASET ===========
class AlgebraicDataset(Dataset):
    def __init__(self, f, interval, nsamples):
        X = urand.Uniform(interval[0], interval[1]).sample([nsamples])
        self.data = [(x, f(x)) for x in X]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ======== CONFIG ===========
line = lambda x: 2 * x + 3
interval = (-10, 10)
train_nsamples = 1000
test_nsamples = 100

train_dataset = AlgebraicDataset(line, interval, train_nsamples)
test_dataset = AlgebraicDataset(line, interval, test_nsamples)

train_dataloader = DataLoader(train_dataset, batch_size=train_nsamples, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=test_nsamples, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

model = LineNetwork().to(device)
lossfunc = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# ======== FUNÇÕES ===========
def train(model, dataloader, lossfunc, optimizer):
    model.train()
    cumloss = 0.0
    for x, y in dataloader:
        x = x.unsqueeze(1).float().to(device)
        y = y.unsqueeze(1).float().to(device)

        pred = model(x)
        loss = lossfunc(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cumloss += loss.item()

    return cumloss / len(dataloader)


def test(model, dataloader, lossfunc):
    model.eval()
    cumloss = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.unsqueeze(1).float().to(device)
            y = y.unsqueeze(1).float().to(device)

            pred = model(x)
            loss = lossfunc(pred, y)
            cumloss += loss.item()

    return cumloss / len(dataloader)


# ======== VISUALIZAÇÃO ===========
def get_plot_data(f, model, interval=(-10, 10), nsamples=10):
    samples = np.linspace(interval[0], interval[1], nsamples)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(samples).unsqueeze(1).float().to(device))
    return samples, list(map(f, samples)), pred.cpu().numpy()


# ======== LOOP DE TREINO ===========
epochs = 1001
plot_epochs = []
plot_data = []

for t in range(epochs):
    train_loss = train(model, train_dataloader, lossfunc, optimizer)
    if t % 50 == 0:
        print(f"Epoch: {t}; Train Loss: {train_loss}")
        samples, ground_truth, pred = get_plot_data(line, model)
        plot_epochs.append(t)
        plot_data.append((samples, ground_truth, pred))

test_loss = test(model, test_dataloader, lossfunc)
print(f"\nTest Loss: {test_loss}")


# ======== MOSTRAR TODAS AS IMAGENS (2 por linha) ===========
n_plots = len(plot_data)
cols = 5
rows = (n_plots + cols - 1) // cols  # arredonda pra cima

fig, axs = plt.subplots(rows, cols, figsize=(10, 5 * rows))
axs = np.array(axs).reshape(-1)  # achata pra iterar fácil

for i, (samples, gt, pred) in enumerate(plot_data):
    ax = axs[i]
    ax.plot(samples, gt, "o", label="Ground Truth")
    ax.plot(samples, pred, label="Model Prediction")
    ax.set_title(f"Epoch {plot_epochs[i]}")
    ax.legend()
    ax.grid(True)

# Remove eixos vazios se sobrar subplot
for j in range(i + 1, len(axs)):
    axs[j].axis("off")

plt.tight_layout()
plt.show()
