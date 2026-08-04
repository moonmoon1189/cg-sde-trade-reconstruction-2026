"""
CG-SDE Sampling for Counterfactual Reconstruction
Reference: Section 4 of the manuscript
"""

import torch
import numpy as np
from models import ConditionalGraphSDE, PoincareManifold


def load_model(model_path: str, config: dict) -> ConditionalGraphSDE:
    """Load trained CG-SDE model"""
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
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def generate_counterfactual_reconstruction(
        model: ConditionalGraphSDE,
        num_nodes: int = 2464,
        node_dim: int = 64,
        device: str = 'cuda'
) -> torch.Tensor:
    """Generate counterfactual reconstruction from pure noise"""
    # Sample pure noise state
    noise_state = torch.randn(1, num_nodes, node_dim).to(device)

    # Apply projection to Poincaré ball
    manifold = PoincareManifold(dim=node_dim)
    noise_state = manifold.project_to_ball(noise_state)

    # Generate condition matrix
    condition = torch.randn_like(noise_state).to(device)

    # Create adjacency matrix
    adjacency = torch.ones(num_nodes, num_nodes).to(device)

    # Reconstruct
    with torch.no_grad():
        reconstructed = model.reconstruct(
            Y_T=noise_state,
            C=condition,
            adjacency=adjacency
        )

    return reconstructed


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        'leaky_slope': 0.2
    }

    model = load_model('cg_sde_model.pt', config)
    model = model.to(device)

    reconstructed = generate_counterfactual_reconstruction(model, device=device)
    print(f"Reconstruction shape: {reconstructed.shape}")
    print("Reconstruction completed successfully.")


if __name__ == '__main__':
    main()