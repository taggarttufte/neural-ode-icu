"""
Latent ODE model for ICU mortality prediction.

Architecture (Rubanova et al. 2019):
  1. RNN Encoder  — processes observations backwards to get z0
  2. Latent ODE   — Neural ODE evolves z0 forward in time
  3. Decoder      — maps final latent state to mortality probability
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint


class ODEFunc(nn.Module):
    """The ODE function dz/dt = f(z, t). Modeled as a small MLP."""

    def __init__(self, latent_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, t, z):
        return self.net(z)


class GRUEncoder(nn.Module):
    """
    GRU-based encoder that processes (value, mask) pairs at each time step
    in reverse order to produce the initial latent state z0.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # Input: observed values + mask for each variable
        self.gru = nn.GRU(input_dim * 2, hidden_dim, batch_first=True)
        self.hidden_to_z0 = nn.Linear(hidden_dim, latent_dim * 2)  # mean + logvar
        self.latent_dim = latent_dim

    def forward(self, values, mask, seq_lengths):
        """
        values : [B, T, N_VARS]
        mask   : [B, T, N_VARS]
        seq_lengths : [B]
        Returns z0_mean, z0_logvar : [B, latent_dim]
        """
        # Concatenate values and mask as input features
        x = torch.cat([values, mask], dim=-1)  # [B, T, N_VARS*2]

        # Reverse the sequence (encode from last obs backwards)
        x_rev = torch.flip(x, dims=[1])

        # Pack for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(
            x_rev, seq_lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        _, hidden = self.gru(packed)  # hidden: [1, B, hidden_dim]
        hidden = hidden.squeeze(0)    # [B, hidden_dim]

        # Project to latent space
        z0_params = self.hidden_to_z0(hidden)  # [B, latent_dim*2]
        z0_mean = z0_params[:, :self.latent_dim]
        z0_logvar = z0_params[:, self.latent_dim:]
        return z0_mean, z0_logvar


class LatentODE(nn.Module):
    """
    Full Latent ODE model:
      Encoder -> sample z0 -> ODE solve -> decode to prediction
    """

    def __init__(self, input_dim, hidden_dim=64, latent_dim=32, ode_hidden_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = GRUEncoder(input_dim, hidden_dim, latent_dim)
        self.ode_func = ODEFunc(latent_dim, ode_hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def reparameterize(self, mean, logvar):
        """Sample z0 using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, times, values, mask, seq_lengths):
        """
        times       : [B, T] — observation times in hours
        values      : [B, T, N_VARS]
        mask        : [B, T, N_VARS]
        seq_lengths : [B]

        Returns:
            logits      : [B] mortality prediction (pre-sigmoid)
            kl_loss     : KL divergence term for VAE training
        """
        B = values.shape[0]

        # Encode to get z0
        z0_mean, z0_logvar = self.encoder(values, mask, seq_lengths)
        z0 = self.reparameterize(z0_mean, z0_logvar)  # [B, latent_dim]

        # Solve ODE from t=0 to t=48 (or just use z0 -> z_final)
        # Use a fixed evaluation time grid for simplicity
        t_span = torch.tensor([0.0, 48.0], device=values.device)
        # odeint expects [T, B, D]
        z_traj = odeint(self.ode_func, z0, t_span, method='dopri5')  # [2, B, latent_dim]
        z_final = z_traj[-1]  # [B, latent_dim] — state at t=48h

        # Decode
        logits = self.decoder(z_final).squeeze(-1)  # [B]

        # KL divergence: KL(q(z0) || N(0,I))
        kl_loss = -0.5 * torch.mean(1 + z0_logvar - z0_mean.pow(2) - z0_logvar.exp())

        return logits, kl_loss

    @torch.no_grad()
    def predict_proba(self, times, values, mask, seq_lengths):
        """Return mortality probability (no sampling noise — use mean z0)."""
        z0_mean, _ = self.encoder(values, mask, seq_lengths)
        t_span = torch.tensor([0.0, 48.0], device=values.device)
        z_traj = odeint(self.ode_func, z0_mean, t_span, method='dopri5')
        z_final = z_traj[-1]
        return torch.sigmoid(self.decoder(z_final).squeeze(-1))
