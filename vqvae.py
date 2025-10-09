import torch
import torch.nn as nn
import torch.nn.functional as F

class VQEmbeddingEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embedding = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", torch.randn(num_embeddings, embedding_dim))
        self.decay = decay
        self.eps = eps

    def forward(self, z_e):
        # Flatten input
        z_e_flat = z_e.permute(0, 2, 3, 1).contiguous()
        z_e_flat = z_e_flat.view(-1, self.embedding_dim)

        distances = (z_e_flat ** 2).sum(1, keepdim=True) - 2 * z_e_flat @ self.embedding.T + (self.embedding ** 2).sum(1)
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()

        quantized = encodings @ self.embedding
        quantized = quantized.view(z_e.shape).permute(0, 3, 1, 2).contiguous()

        if self.training:
            enc_sum = encodings.sum(0)
            self.ema_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * enc_sum

            dw = encodings.T @ z_e_flat
            self.ema_w = self.decay * self.ema_w + (1 - self.decay) * dw

            n = self.ema_cluster_size.sum()
            self.embedding.data = self.ema_w / self.ema_cluster_size.unsqueeze(1).clamp(min=self.eps)

        B, D, Hq, Wq = z_e.shape
        encoding_indices = encoding_indices.view(B, Hq, Wq)
        return quantized, (quantized - z_e).detach() + z_e, encoding_indices

class VQVAE(nn.Module):
    def __init__(self, in_channels=6, hidden_dim=128, embedding_dim=64, num_embeddings=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, embedding_dim, 3, padding=1)
        )
        self.vq = VQEmbeddingEMA(num_embeddings, embedding_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, in_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, z_q_straight, _ = self.vq(z_e)
        x_recon = self.decoder(z_q_straight)
        return x_recon, z_e, z_q

def generate_ensemble(model, vqvae, x_tm2, x_tm1, steps=6, members=10):
    forecasts = []
    for _ in range(members):
        z_prev2, z_prev1 = encode_vq(x_tm2), encode_vq(x_tm1)
        pred_seq = []
        for t in range(steps):
            z_input = torch.cat([z_prev2, z_prev1], dim=1)
            z_t = torch.randn_like(z_prev1)
            x_input = torch.cat([z_input, z_t], dim=1)
            pred_noise = model(x_input, t=torch.tensor([0.5]))  # or use schedule sampling
            z_q = z_t - pred_noise * ...
            pred_frame = vqvae.decoder(z_q)
            pred_seq.append(pred_frame)
            z_prev2, z_prev1 = z_prev1, encode_vq(pred_frame)
        forecasts.append(torch.stack(pred_seq, dim=1))
    return torch.stack(forecasts, dim=0)