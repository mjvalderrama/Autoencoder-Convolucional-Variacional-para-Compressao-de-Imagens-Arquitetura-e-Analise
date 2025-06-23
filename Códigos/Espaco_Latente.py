import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch import nn

# Definir uso de CPU (se tiver GPU configurada, pode usar "cuda")
device = torch.device("cpu")

# Carregar MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(mnist_data, batch_size=512, shuffle=False)

# Definir Autoencoder simples com espaço latente 2D
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Latent space 2D
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

# Instanciar e treinar
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(100):
    for batch_x, _ in data_loader:
        batch_x = batch_x.to(device)
        optimizer.zero_grad()
        x_recon, _ = model(batch_x)
        loss = criterion(x_recon, batch_x)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Obter as representações no espaço latente
model.eval()
all_z = []
all_labels = []

with torch.no_grad():
    for batch_x, batch_labels in data_loader:
        batch_x = batch_x.to(device)
        _, z = model(batch_x)
        all_z.append(z.cpu().numpy())
        all_labels.append(batch_labels.numpy())


# Após coletar todas as saídas do encoder
all_z = np.concatenate(all_z, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Limitar número de pontos
max_points = 500
indices = np.random.choice(len(all_z), max_points, replace=False)
all_z_plot = all_z[indices]
all_labels_plot = all_labels[indices]

# Plotar
plt.figure(figsize=(8, 6))
scatter = plt.scatter(all_z_plot[:, 0], all_z_plot[:, 1], c=all_labels_plot, cmap='tab10', s=20)
plt.colorbar(scatter, ticks=range(10))
plt.title('Espaço Latente 2D - Autoencoder MNIST (Amostra Aleatória de 500 pontos)')
plt.xlabel('Dimensão Latente 1')
plt.ylabel('Dimensão Latente 2')
plt.tight_layout()
plt.show()