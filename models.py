"""
CG-SDE: Conditional Graph Stochastic Differential Equation
Global Trade Disturbance Propagation Path Reconstruction
Reference: Section 2.2-2.4 of the manuscript
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Optional, Dict, List, Callable
import math
import warnings

warnings.filterwarnings('ignore')


class PoincareManifold:
    """Poincaré ball model operations for hyperbolic geometry"""

    def __init__(self, dim: int = 64, c: float = 1.0):
        self.dim = dim
        self.c = c

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Möbius addition on Poincaré ball"""
        x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y ** 2, dim=-1, keepdim=True)
        xy_dot = torch.sum(x * y, dim=-1, keepdim=True)

        numerator = (1 + 2 * self.c * xy_dot + self.c * y_norm_sq) * x + (1 - self.c * x_norm_sq) * y
        denominator = 1 + 2 * self.c * xy_dot + self.c ** 2 * x_norm_sq * y_norm_sq

        return numerator / (denominator + 1e-8)

    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Riemannian exponential map from tangent space to manifold"""
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        v_norm = torch.norm(v, dim=-1, keepdim=True)

        lambda_x = 2 / (1 - self.c * x_norm ** 2 + 1e-8)
        v_scaled = v * lambda_x

        v_scaled_norm = torch.norm(v_scaled, dim=-1, keepdim=True)
        tanh_factor = torch.tanh(torch.sqrt(torch.tensor(self.c)) * v_scaled_norm / 2)

        direction = v_scaled / (v_scaled_norm + 1e-8)
        result = self.mobius_add(x, direction * tanh_factor / torch.sqrt(torch.tensor(self.c)))
        return result

    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Riemannian logarithmic map from manifold to tangent space"""
        diff = self.mobius_add(-x, y)
        diff_norm = torch.norm(diff, dim=-1, keepdim=True)
        lambda_x = 2 / (1 - self.c * torch.norm(x, dim=-1, keepdim=True) ** 2 + 1e-8)

        norm_factor = 2 / (torch.sqrt(torch.tensor(self.c)) * lambda_x)
        atanh_factor = torch.atanh(torch.sqrt(torch.tensor(self.c)) * diff_norm)

        return diff * (norm_factor * atanh_factor / (diff_norm + 1e-8))

    def geodesic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Hyperbolic geodesic distance between points"""
        diff = self.mobius_add(-x, y)
        diff_norm = torch.norm(diff, dim=-1)
        return 2 / torch.sqrt(torch.tensor(self.c)) * torch.atanh(torch.sqrt(torch.tensor(self.c)) * diff_norm)

    def project_to_ball(self, x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Project tensor onto Poincaré ball with numerical stability"""
        norm = torch.norm(x, dim=-1, keepdim=True)
        max_norm = 1.0 / torch.sqrt(torch.tensor(self.c)) - eps
        scale = torch.clamp(max_norm / (norm + eps), max=1.0)
        return x * scale


class ScoreGNN(nn.Module):
    """Multi-order spatial attention score matching network"""

    def __init__(
            self,
            node_dim: int = 64,
            hidden_dim: int = 256,
            num_layers: int = 4,
            M: int = 3,
            dropout: float = 0.15,
            leaky_slope: float = 0.2
    ):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.M = M
        self.dropout = dropout

        self.W_q = nn.Linear(node_dim, hidden_dim)
        self.W_k = nn.Linear(node_dim, hidden_dim)
        self.W_c = nn.Linear(node_dim, hidden_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        self.projection_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.projection_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(leaky_slope),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            )

        self.output_proj = nn.Linear(hidden_dim, node_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.manifold = PoincareManifold(dim=node_dim)

    def forward(
            self,
            Y: torch.Tensor,
            C: torch.Tensor,
            t: torch.Tensor,
            adjacency: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through Score-GNN"""
        batch_size, num_nodes, feat_dim = Y.shape

        # Project input features
        Z = self.W_q(Y)

        # Multi-order message passing
        for m in range(1, self.M + 1):
            m_hop_adj = torch.matrix_power(adjacency.float(), m)
            m_hop_adj = (m_hop_adj > 0).float()

            for l in range(self.num_layers):
                attn_out, _ = self.attention(Z, Z, Z, key_padding_mask=None)
                attn_out = attn_out + Z
                attn_out = self.layer_norm(attn_out)

                proj_out = self.projection_layers[l](attn_out)
                Z = Z + proj_out
                Z = self.layer_norm(Z)
                Z = F.dropout(Z, p=self.dropout, training=self.training)

        gradient_field = self.output_proj(Z)

        # Apply Riemannian gradient scaling (Equation 6)
        y_norm_sq = torch.sum(Y ** 2, dim=-1, keepdim=True)
        scale = ((1 - y_norm_sq) ** 2) / 4
        gradient_field = gradient_field * scale

        return gradient_field


class ForwardSDE:
    """Forward stochastic differential equation degradation chain"""

    def __init__(
            self,
            theta_min: float = 0.1,
            theta_max: float = 20.0,
            T: float = 1.0,
            manifold: Optional[PoincareManifold] = None
    ):
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.T = T
        self.manifold = manifold or PoincareManifold()

    def noise_schedule(self, t: torch.Tensor) -> torch.Tensor:
        """Linear noise scheduling function"""
        return self.theta_min + t * (self.theta_max - self.theta_min)

    def forward_step(
            self,
            Y: torch.Tensor,
            t: torch.Tensor,
            dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single forward diffusion step"""
        theta_t = self.noise_schedule(t)
        drift = -0.5 * theta_t * Y * dt
        noise = torch.randn_like(Y) * torch.sqrt(theta_t * dt)
        Y_next = Y + drift + noise
        Y_next = self.manifold.project_to_ball(Y_next)
        return Y_next, noise

    def forward_chain(
            self,
            Y_0: torch.Tensor,
            K: int = 1000
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward degradation chain"""
        dt = self.T / K
        Y_t = Y_0.clone()
        noise_history = []

        for k in range(K):
            t_k = torch.tensor(k * dt)
            Y_t, noise = self.forward_step(Y_t, t_k, dt)
            noise_history.append(noise)

        return Y_t, torch.stack(noise_history)


class InverseEulerMaruyamaSolver:
    """Reverse-time Euler-Maruyama solver for counterfactual reconstruction"""

    def __init__(
            self,
            K: int = 1000,
            theta_min: float = 0.1,
            theta_max: float = 20.0,
            T: float = 1.0,
            manifold: Optional[PoincareManifold] = None
    ):
        self.K = K
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.T = T
        self.dt = T / K
        self.manifold = manifold or PoincareManifold()

    def noise_schedule(self, t: torch.Tensor) -> torch.Tensor:
        return self.theta_min + t * (self.theta_max - self.theta_min)

    def mass_preserving_projection(
            self,
            Y: torch.Tensor,
            b_i: torch.Tensor,
            eps_i: torch.Tensor
    ) -> torch.Tensor:
        """Dykstra alternating projection for flow conservation (Equation 9-10)"""
        inflow = torch.sum(Y, dim=1, keepdim=True)
        outflow = torch.sum(Y, dim=0, keepdim=True)
        imbalance = inflow - outflow - b_i

        mask = torch.abs(imbalance) > eps_i
        adjustment = torch.where(
            mask,
            (imbalance - torch.sign(imbalance) * eps_i) / 2,
            torch.zeros_like(imbalance)
        )

        Y_proj = Y - adjustment * mask.float()
        return Y_proj

    def solve(
            self,
            Y_T: torch.Tensor,
            C: torch.Tensor,
            score_func: Callable,
            adjacency: torch.Tensor,
            b_i: Optional[torch.Tensor] = None,
            eps_i: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Inverse Euler-Maruyama denoising reconstruction (Equation 8)"""
        Y_k = Y_T.clone()

        for k in range(self.K, 0, -1):
            t_k = torch.tensor(k * self.dt)
            theta_tk = self.noise_schedule(t_k)

            xi_k = torch.randn_like(Y_k) * torch.sqrt(theta_tk * self.dt)
            score = score_func(Y_k, C, t_k, adjacency)

            drift_correction = (
                    0.5 * theta_tk * Y_k * self.dt
                    + theta_tk * score * self.dt
            )
            Y_prev = Y_k + drift_correction + xi_k

            Y_prev = self.manifold.project_to_ball(Y_prev)

            if b_i is not None and eps_i is not None:
                Y_prev = self.mass_preserving_projection(Y_prev, b_i, eps_i)

            Y_k = Y_prev

        return Y_k


class ConditionalGraphSDE(nn.Module):
    """Complete CG-SDE framework for trade network reconstruction"""

    def __init__(
            self,
            node_dim: int = 64,
            hidden_dim: int = 256,
            num_layers: int = 4,
            M: int = 3,
            K: int = 1000,
            theta_min: float = 0.1,
            theta_max: float = 20.0,
            T: float = 1.0,
            dropout: float = 0.15,
            leaky_slope: float = 0.2
    ):
        super().__init__()

        self.node_dim = node_dim
        self.K = K
        self.T = T
        self.dt = T / K

        self.manifold = PoincareManifold(dim=node_dim)
        self.score_gnn = ScoreGNN(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            M=M,
            dropout=dropout,
            leaky_slope=leaky_slope
        )
        self.forward_sde = ForwardSDE(
            theta_min=theta_min,
            theta_max=theta_max,
            T=T,
            manifold=self.manifold
        )
        self.inverse_solver = InverseEulerMaruyamaSolver(
            K=K,
            theta_min=theta_min,
            theta_max=theta_max,
            T=T,
            manifold=self.manifold
        )

    def compute_loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Time-dependent loss balancing weight lambda(t)"""
        theta_t = self.forward_sde.noise_schedule(t)
        return theta_t

    def forward_denoising_loss(
            self,
            Y_0: torch.Tensor,
            C: torch.Tensor,
            t: torch.Tensor,
            adjacency: torch.Tensor
    ) -> torch.Tensor:
        """Structured denoising score matching loss (Equation 5)"""
        Y_t, noise = self.forward_sde.forward_step(Y_0, t, self.dt)

        score_pred = self.score_gnn(Y_t, C, t, adjacency)

        theta_t = self.forward_sde.noise_schedule(t)
        score_true = -noise / (torch.sqrt(theta_t * self.dt) + 1e-8)

        y_norm_sq = torch.sum(Y_t ** 2, dim=-1, keepdim=True)
        scale = ((1 - y_norm_sq) ** 2) / 4
        score_true = score_true * scale

        lambda_t = self.compute_loss_weight(t)
        loss = lambda_t * torch.mean((score_pred - score_true) ** 2)

        return loss

    def train_step(
            self,
            Y_0: torch.Tensor,
            C: torch.Tensor,
            adjacency: torch.Tensor
    ) -> torch.Tensor:
        t = torch.rand(1).to(Y_0.device) * self.T
        loss = self.forward_denoising_loss(Y_0, C, t, adjacency)
        return loss

    def reconstruct(
            self,
            Y_T: torch.Tensor,
            C: torch.Tensor,
            adjacency: torch.Tensor,
            b_i: Optional[torch.Tensor] = None,
            eps_i: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        score_func = self.score_gnn
        Y_0_reconstructed = self.inverse_solver.solve(
            Y_T=Y_T,
            C=C,
            score_func=score_func,
            adjacency=adjacency,
            b_i=b_i,
            eps_i=eps_i
        )
        return Y_0_reconstructed


class TradeNetworkDataset(Dataset):
    """Dataset for temporal trade graph snapshots"""

    def __init__(
            self,
            tensor_path: str,
            time_indices: List[int],
            node_dim: int = 64
    ):
        self.tensor_path = tensor_path
        self.time_indices = time_indices
        self.node_dim = node_dim

        # Load preprocessed tensor
        self.data = np.load(tensor_path)
        self.num_nodes = self.data.shape[1]

        if self.data.ndim == 3:
            self.num_time_steps = self.data.shape[0]
            self.num_nodes = self.data.shape[1]
            self.feat_dim = self.data.shape[2]

    def __len__(self):
        return len(self.time_indices)

    def __getitem__(self, idx):
        t_idx = self.time_indices[idx]
        Y_t = torch.tensor(self.data[t_idx, :, :self.node_dim], dtype=torch.float32)
        adjacency = torch.ones(self.num_nodes, self.num_nodes, dtype=torch.float32)
        return Y_t, adjacency


def train_model(
        train_loader: DataLoader,
        model: ConditionalGraphSDE,
        epochs: int = 5000,
        lr: float = 2e-4,
        weight_decay: float = 1e-4,
        shadow_decay: float = 0.999,
        device: str = 'cuda'
):
    """Training loop with shadow update mechanism"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    shadow_params = {
        name: param.data.clone()
        for name, param in model.named_parameters()
    }

    for epoch in range(epochs):
        epoch_loss = 0.0

        for Y_0, adjacency in train_loader:
            Y_0 = Y_0.to(device)
            adjacency = adjacency.to(device)
            C = torch.randn_like(Y_0).to(device)

            loss = model.train_step(Y_0, C, adjacency)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        if epoch % 500 == 0 and epoch > 0:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    shadow_params[name] = (
                            shadow_decay * shadow_params[name]
                            + (1 - shadow_decay) * param.data
                    )
            print(f"Epoch {epoch}, Loss: {epoch_loss / len(train_loader):.6f}")

    with torch.no_grad():
        for name, param in model.named_parameters():
            param.data = shadow_params[name]

    return model


def main_training():
    """Main training entry point"""
    config = {
        'node_dim': 64,
        'hidden_dim': 256,
        'num_layers': 4,
        'M': 3,
        'K': 1000,
        'theta_min': 0.1,
        'theta_max': 20.0,
        'T': 1.0,
        'dropout': 0.15,
        'leaky_slope': 0.2,
        'batch_size': 16,
        'epochs': 5000,
        'lr': 2e-4,
        'weight_decay': 1e-4,
        'shadow_decay': 0.999,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Create synthetic data for testing
    dummy_data = np.random.randn(100, 64, config['node_dim']).astype(np.float32)
    np.save('dummy_trade_tensor.npy', dummy_data)

    dataset = TradeNetworkDataset(
        tensor_path='dummy_trade_tensor.npy',
        time_indices=list(range(100)),
        node_dim=config['node_dim']
    )
    train_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )

    model = ConditionalGraphSDE(
        node_dim=config['node_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        M=config['M'],
        K=config['K'],
        theta_min=config['theta_min'],
        theta_max=config['theta_max'],
        T=config['T'],
        dropout=config['dropout'],
        leaky_slope=config['leaky_slope']
    )

    model = train_model(
        train_loader=train_loader,
        model=model,
        epochs=config['epochs'],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        shadow_decay=config['shadow_decay'],
        device=config['device']
    )

    torch.save(model.state_dict(), 'cg_sde_model.pt')
    print("Model saved successfully.")

    return model


def sample_reconstruction(
        model: ConditionalGraphSDE,
        noise_state: torch.Tensor,
        condition: torch.Tensor,
        adjacency: torch.Tensor,
        b_i: Optional[torch.Tensor] = None,
        eps_i: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Sampling function for counterfactual reconstruction"""
    model.eval()
    with torch.no_grad():
        reconstructed = model.reconstruct(
            Y_T=noise_state,
            C=condition,
            adjacency=adjacency,
            b_i=b_i,
            eps_i=eps_i
        )
    return reconstructed


if __name__ == '__main__':
    model = main_training()
    print("Training completed successfully.")