"""
Physics-Informed Constraints for N-Body Simulation

This module provides functions to compute momentum for charged particle
systems, enabling physics-informed loss terms that encourage conservation laws.

Conservation Laws:
    - Linear Momentum: P = sum(m_i * v_i) should be conserved (no external forces)
"""

import torch


def compute_linear_momentum(vel, masses=None):
    """
    Compute total linear momentum: P = sum(m_i * v_i)
    
    For an isolated system, total momentum should be conserved.
    
    Args:
        vel: Velocities [batch_size, n_nodes, 3] or [batch_size * n_nodes, 3]
        masses: Optional masses. If None, assumes unit mass.
    
    Returns:
        Total momentum vector per sample [batch_size, 3] or [3] if unbatched
    """
    if vel.dim() == 3:
        # vel is [batch_size, n_nodes, 3]
        if masses is not None:
            momentum = (vel * masses.unsqueeze(-1)).sum(dim=1)
        else:
            momentum = vel.sum(dim=1)  # [batch_size, 3]
    else:
        # vel is [batch_size * n_nodes, 3] - caller needs to reshape
        if masses is not None:
            momentum = (vel * masses.unsqueeze(-1)).sum(dim=0)
        else:
            momentum = vel.sum(dim=0)  # [3]
    
    return momentum


def compute_linear_momentum_batched(vel, n_nodes, batch_size, masses=None):
    """
    Compute total linear momentum for batched data.
    
    Args:
        vel: Velocities [batch_size * n_nodes, 3]
        n_nodes: Number of nodes per sample
        batch_size: Batch size
        masses: Optional masses
    
    Returns:
        Total momentum per sample [batch_size, 3]
    """
    vel_batched = vel.view(batch_size, n_nodes, 3)
    return compute_linear_momentum(vel_batched, masses)


def momentum_conservation_loss(vel_init, vel_pred, n_nodes, batch_size, masses=None):
    """
    Compute momentum conservation loss: penalize deviation from initial momentum.
    
    Loss = mean(|P_pred - P_init|^2)
    
    Args:
        vel_init: Initial velocities [batch_size * n_nodes, 3]
        vel_pred: Predicted velocities [batch_size * n_nodes, 3]
        n_nodes: Number of nodes
        batch_size: Batch size
        masses: Optional masses
    
    Returns:
        Scalar momentum conservation loss
    """
    P_init = compute_linear_momentum_batched(vel_init, n_nodes, batch_size, masses)
    P_pred = compute_linear_momentum_batched(vel_pred, n_nodes, batch_size, masses)
    
    # L2 loss on momentum difference
    loss = ((P_pred - P_init) ** 2).sum(dim=-1)  # [batch_size]
    
    # Normalize by initial momentum magnitude
    P_init_norm = (P_init ** 2).sum(dim=-1).clamp(min=1e-6)
    loss = loss / P_init_norm
    
    return loss.mean()


class PhysicsInformedLoss:
    """
    Wrapper class for physics-informed loss computation.
    
    Enforces momentum conservation constraint.
    """
    
    def __init__(self, lambda_momentum=0.1, warmup_epochs=0, max_physics_loss=10.0):
        """
        Args:
            lambda_momentum: Weight for momentum conservation loss
            warmup_epochs: Number of epochs to linearly ramp up physics loss
            max_physics_loss: Maximum value to clamp physics loss (prevents domination)
        """
        self.lambda_momentum = lambda_momentum
        self.warmup_epochs = warmup_epochs
        self.max_physics_loss = max_physics_loss
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        """Update current epoch for warmup scheduling."""
        self.current_epoch = epoch
    
    def get_warmup_factor(self):
        """Get the warmup scaling factor (0 to 1)."""
        if self.warmup_epochs <= 0:
            return 1.0
        return min(1.0, self.current_epoch / self.warmup_epochs)
    
    def compute_physics_losses(self, vel_init, vel_pred, n_nodes, batch_size, masses=None):
        """
        Compute momentum conservation loss.
        
        Args:
            vel_init: Initial velocities
            vel_pred: Predicted velocities
            n_nodes: Number of nodes per sample
            batch_size: Batch size
            masses: Optional masses
        
        Returns:
            Tuple of (total_physics_loss, dict with individual loss components)
        """
        losses = {}
        total_physics_loss = 0.0
        warmup_factor = self.get_warmup_factor()
        
        if self.lambda_momentum > 0:
            momentum_loss = momentum_conservation_loss(
                vel_init, vel_pred, n_nodes, batch_size, masses
            )
            # Clamp to prevent extreme values
            momentum_loss_clamped = torch.clamp(momentum_loss, max=self.max_physics_loss / max(self.lambda_momentum, 1e-6))
            losses['momentum'] = momentum_loss.item()  # Log unclamped for monitoring
            losses['momentum_clamped'] = momentum_loss_clamped.item()
            total_physics_loss = total_physics_loss + self.lambda_momentum * momentum_loss_clamped * warmup_factor
        
        losses['total_physics'] = total_physics_loss.item() if torch.is_tensor(total_physics_loss) else total_physics_loss
        losses['warmup_factor'] = warmup_factor
        
        return total_physics_loss, losses
