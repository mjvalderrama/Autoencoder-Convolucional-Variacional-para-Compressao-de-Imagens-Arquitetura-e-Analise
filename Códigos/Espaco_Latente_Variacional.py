import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# ----------------------------
# Definição do VAE
# ----------------------------
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 400),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(400, 2)
        self.fc_logvar = nn.Linear(400, 2)

        self.decoder = nn.Sequential(
            nn.Linear(2, 400),
            nn.ReLU(),
            nn.Linear(400, 28 * 28),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# ----------------------------
# Função de Loss
# ----------------------------
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# ----------------------------
# Carregar MNIST
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# ----------------------------
# Treinamento
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 50
model.train()
for epoch in range(epochs):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset):.4f}')

# ----------------------------
# Espaço Latente: Extração dos mus e logvars
# ----------------------------
model.eval()
latents = []
labels = []

with torch.no_grad():
    for i in range(2500):  # Reduzido para facilitar a visualização
        img, label = train_dataset[i]
        img = img.to(device)
        mu, logvar = model.encode(img.unsqueeze(0))
        latents.append((mu.cpu().numpy().flatten(), logvar.cpu().numpy().flatten()))
        labels.append(label)

mus = np.array([l[0] for l in latents])
logvars = np.array([l[1] for l in latents])
vars_ = np.exp(logvars)

# ----------------------------
# Plot com Elipses e Colorbar
# ----------------------------
plt.figure(figsize=(10, 8))
colors = plt.cm.tab10(np.linspace(0, 1, 10))

# Plot das elipses (variâncias)
for i in range(len(mus)):
    x, y = mus[i]
    vx, vy = vars_[i]
    color = colors[labels[i]]
    ellipse = Ellipse((x, y), width=2 * np.sqrt(vx), height=2 * np.sqrt(vy),
                      edgecolor=color, facecolor='none', linewidth=0.5, alpha=0.5)
    plt.gca().add_patch(ellipse)

# Plot dos pontos (mus)
scatter = plt.scatter(mus[:, 0], mus[:, 1], c=labels, cmap='tab10', s=1, alpha=0.5)

# Títulos e eixos
plt.title('Espaço Latente 2D do VAE (com variâncias)')
plt.xlabel('Dimensão Latente 1')
plt.ylabel('Dimensão Latente 2')
plt.grid(False)

# Colorbar lateral
cbar = plt.colorbar(scatter, ticks=range(10))
cbar.set_label('Classe (Dígito MNIST)')
cbar.set_ticks(range(10))
cbar.set_ticklabels([str(i) for i in range(10)])

plt.tight_layout()
plt.show()
