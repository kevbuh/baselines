# Paper: https://arxiv.org/abs/1711.00937
# Implementation: https://github.com/ariG23498/variational-image-models/blob/main/vector_quantised_variational_autoencoder_cifar10.ipynb

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

batch_size = 128
learning_rate = 1e-3
num_epochs = 5
image_size = 32
channels = 3
latent_dim = 128
num_embeddings = 512  # Number of vectors in the codebook
commitment_cost = 0.25  # Beta, the commitment loss weight
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VQEmbedding(nn.Module):
  def __init__(self, num_embeddings, embedding_dim):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.embedding = nn.Embedding(num_embeddings, embedding_dim)
    # uniform init prevents bias 
    self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

  def forward(self, z):
    b, c, h, w = z.shape
    z_channel_last = z.permute(0, 2, 3, 1) # [b, h, w, c]
    z_flattened = z_channel_last.reshape(b*h*w, self.embedding_dim) # [bhw, embedding_dim]
    # MSE = ||z-z_q||^2 = (z^2)+(z_q^2)-(2zz_q^T)
    # shape = [b*h*w, num_embeddings]
    distances = (
      torch.sum(z_flattened**2, dim=-1, keepdim=True)
      + torch.sum(self.embedding.weight.t() ** 2, dim=0, keepdim=True)
      - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
    )
    encoding_indices = torch.argmin(distances, dim=-1)
    # pluck and revert shape to [b, embedding_dim, h, w]
    z_q = self.embedding(encoding_indices)
    z_q = z_q.reshape(b, h, w, self.embedding_dim)
    z_q = z_q.permute(0, 3, 1, 2)
    # commitment loss
    loss = F.mse_loss(z_q, z.detach()) + commitment_cost * F.mse_loss(z_q.detach(), z)
    # straight through estimator
    z_q = z + (z_q - z).detach()
    return z_q, loss, encoding_indices
    
class VQVAE(nn.Module):
  def __init__(self):
    super(VQVAE, self).__init__()
    self.encoder = nn.Sequential(
      nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
      nn.ReLU(),
      nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
      nn.ReLU(),
      nn.Conv2d(128, latent_dim, kernel_size=1)
    )
    self.vq_layer = VQEmbedding(num_embeddings, latent_dim)
    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(latent_dim, 128, kernel_size=1),
      nn.ReLU(),
      nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
      nn.ReLU(),
      nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
      nn.ReLU(),
      nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1),
      nn.Tanh()
    )

  def forward(self, x):
    z_e = self.encoder(x)
    z_q, vq_loss, _ = self.vq_layer(z_e)
    x_recon = self.decoder(z_q)
    return x_recon, vq_loss

def vqvae_loss(recon_x, x, vq_loss):
  recon_loss = F.mse_loss(recon_x, x)
  return recon_loss + vq_loss

# training
if __name__ == "__main__":
  import matplotlib.pyplot as plt
  import torchvision
  from tqdm import tqdm
  
  def show_image(batch_of_tensors):
    images = batch_of_tensors[:4]
    images = (images * 0.5) + 0.5  # Unnormalize the images to [0, 1] range
    grid_img = torchvision.utils.make_grid(images, nrow=2)
    plt.figure(figsize=(5, 5))
    plt.imshow(grid_img.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    plt.axis('off')
    plt.show()

  model = VQVAE().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
  ])

  train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
  train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

  for epoch in tqdm(range(num_epochs)):
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
      data = data.to(device)
      optimizer.zero_grad()
      recon_batch, vq_loss = model(data)
      loss = vqvae_loss(recon_batch, data, vq_loss)
      loss.backward()
      train_loss += loss.item()
      optimizer.step()

    avg_loss = train_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch + 1}/{num_epochs}] Average Loss: {avg_loss:.4f}')

    if epoch % 5 == 0:
      with torch.no_grad():
        recon_batch, _ = model(data)
        show_image(data.cpu())
        show_image(recon_batch.cpu())
