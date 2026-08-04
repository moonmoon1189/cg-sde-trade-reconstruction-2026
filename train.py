"""
CG-SDE Training Entry Point
Reference: Section 3.2 of the manuscript
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from models import ConditionalGraphSDE, TradeNetworkDataset, train_model


def main():
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

    # Load data
    dataset = TradeNetworkDataset(
        tensor_path='data/wiod_daily_interpolated.npy',
        time_indices=list(range(2000, 2011)),
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


if __name__ == '__main__':
    main()