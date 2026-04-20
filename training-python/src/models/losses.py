"""
=============================================================================
XRAYVISION - Loss Functions
=============================================================================
- Focal Loss pour classification (gère le déséquilibre de classes)
- Smooth L1 Loss pour régression bbox
- Combined SSD Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ============================================================================
# FOCAL LOSS
# ============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss pour classification avec déséquilibre de classes
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Poids pour les classes positives (default: 0.25)
        gamma: Facteur de focalisation (default: 2.0)
        reduction: 'none', 'mean', 'sum'
    """
    def __init__(
        self, 
        alpha: float = 0.25, 
        gamma: float = 2.0, 
        reduction: str = 'mean',
        num_classes: int = 17
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            inputs: (N, num_classes) logits
            targets: (N,) class indices
            mask: (N,) optional mask for valid samples
        
        Returns:
            Focal loss value
        """
        # Softmax probabilities
        p = F.softmax(inputs, dim=-1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # Get probability for true class
        p_t = (p * targets_one_hot).sum(dim=-1)
        
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Cross entropy
        ce = F.cross_entropy(inputs, targets, reduction='none')
        
        # Focal loss
        loss = self.alpha * focal_weight * ce
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / mask.sum().clamp(min=1)
            elif self.reduction == 'sum':
                return loss.sum()
            return loss
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ============================================================================
# SMOOTH L1 LOSS (HUBER LOSS)
# ============================================================================
class SmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss pour régression bbox
    
    L(x) = 0.5 * x^2           if |x| < beta
         = |x| - 0.5 * beta    otherwise
    
    Args:
        beta: Seuil de transition (default: 1.0)
        reduction: 'none', 'mean', 'sum'
    """
    def __init__(self, beta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            inputs: (N, 4) predicted bbox offsets
            targets: (N, 4) target bbox offsets
            mask: (N,) optional mask for valid samples
        
        Returns:
            Smooth L1 loss value
        """
        diff = torch.abs(inputs - targets)
        
        # Smooth L1
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        
        # Sum over bbox dimensions
        loss = loss.sum(dim=-1)
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / mask.sum().clamp(min=1)
            elif self.reduction == 'sum':
                return loss.sum()
            return loss
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ============================================================================
# SSD COMBINED LOSS
# ============================================================================
class SSDLoss(nn.Module):
    """
    Combined loss pour SSD: Focal Loss + Smooth L1
    
    L = L_cls + lambda * L_reg
    
    Args:
        num_classes: Nombre de classes
        alpha: Alpha pour Focal Loss
        gamma: Gamma pour Focal Loss
        reg_weight: Poids pour la loss de régression
        neg_pos_ratio: Ratio negative/positive pour hard negative mining
    """
    def __init__(
        self,
        num_classes: int = 17,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reg_weight: float = 1.0,
        neg_pos_ratio: float = 3.0,
        background_class: int = -1  # -1 = pas de classe background explicite
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.reg_weight = reg_weight
        self.neg_pos_ratio = neg_pos_ratio
        self.background_class = background_class
        
        self.cls_loss = FocalLoss(
            alpha=alpha, 
            gamma=gamma, 
            reduction='none',
            num_classes=num_classes
        )
        self.reg_loss = SmoothL1Loss(beta=1.0, reduction='none')
    
    def forward(
        self,
        cls_preds: torch.Tensor,
        reg_preds: torch.Tensor,
        cls_targets: torch.Tensor,
        reg_targets: torch.Tensor,
        pos_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            cls_preds: (B, N, num_classes) classification predictions
            reg_preds: (B, N, 4) regression predictions
            cls_targets: (B, N) target class labels
            reg_targets: (B, N, 4) target bbox offsets
            pos_mask: (B, N) mask for positive samples
        
        Returns:
            total_loss, cls_loss, reg_loss
        """
        batch_size = cls_preds.size(0)
        num_anchors = cls_preds.size(1)
        
        # Flatten
        cls_preds_flat = cls_preds.view(-1, self.num_classes)
        reg_preds_flat = reg_preds.view(-1, 4)
        cls_targets_flat = cls_targets.view(-1)
        reg_targets_flat = reg_targets.view(-1, 4)
        pos_mask_flat = pos_mask.view(-1)
        
        # Nombre de positifs
        num_pos = pos_mask_flat.sum().item()
        
        if num_pos == 0:
            # Pas de positifs, retourne 0
            zero = torch.tensor(0.0, device=cls_preds.device, requires_grad=True)
            return zero, zero, zero
        
        # =====================
        # Classification Loss
        # =====================
        # Hard Negative Mining
        neg_mask = self._hard_negative_mining(
            cls_preds_flat, 
            cls_targets_flat, 
            pos_mask_flat
        )
        
        # Mask combiné (positifs + hard negatives)
        combined_mask = pos_mask_flat | neg_mask
        
        # Focal loss sur les samples sélectionnés
        cls_loss_val = self.cls_loss(
            cls_preds_flat[combined_mask], 
            cls_targets_flat[combined_mask]
        ).mean()
        
        # =====================
        # Regression Loss
        # =====================
        # Seulement sur les positifs
        reg_loss_val = self.reg_loss(
            reg_preds_flat[pos_mask_flat],
            reg_targets_flat[pos_mask_flat]
        ).mean()
        
        # =====================
        # Total Loss
        # =====================
        total_loss = cls_loss_val + self.reg_weight * reg_loss_val
        
        return total_loss, cls_loss_val, reg_loss_val
    
    def _hard_negative_mining(
        self,
        cls_preds: torch.Tensor,
        cls_targets: torch.Tensor,
        pos_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Hard Negative Mining pour équilibrer positifs/négatifs
        """
        num_pos = pos_mask.sum().item()
        
        if num_pos == 0:
            # Si pas de positifs, prendre un échantillon aléatoire
            num_neg = min(int(self.neg_pos_ratio * 10), len(pos_mask))
            neg_mask = torch.zeros_like(pos_mask)
            indices = torch.randperm(len(pos_mask))[:num_neg]
            neg_mask[indices] = True
            return neg_mask
        
        # Nombre de négatifs à sélectionner
        num_neg = int(self.neg_pos_ratio * num_pos)
        
        # Calculer les losses pour les négatifs
        with torch.no_grad():
            loss_all = F.cross_entropy(cls_preds, cls_targets, reduction='none')
            loss_neg = loss_all.clone()
            loss_neg[pos_mask] = 0  # Ignorer les positifs
            
            # Trier par loss décroissante
            _, neg_idx = loss_neg.sort(descending=True)
            
            # Sélectionner les top-k négatifs
            neg_mask = torch.zeros_like(pos_mask)
            neg_mask[neg_idx[:num_neg]] = True
            neg_mask = neg_mask & ~pos_mask  # Exclure les positifs
        
        return neg_mask


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def create_ssd_loss(
    num_classes: int = 17,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reg_weight: float = 1.0
) -> SSDLoss:
    """Factory function pour créer la loss SSD"""
    return SSDLoss(
        num_classes=num_classes,
        alpha=alpha,
        gamma=gamma,
        reg_weight=reg_weight
    )


if __name__ == "__main__":
    # Test des loss functions
    print("🧪 Test des loss functions...")
    
    # Test Focal Loss
    focal = FocalLoss(num_classes=17)
    inputs = torch.randn(100, 17)
    targets = torch.randint(0, 17, (100,))
    fl = focal(inputs, targets)
    print(f"   Focal Loss: {fl.item():.4f}")
    
    # Test Smooth L1
    smooth_l1 = SmoothL1Loss()
    preds = torch.randn(100, 4)
    targs = torch.randn(100, 4)
    sl1 = smooth_l1(preds, targs)
    print(f"   Smooth L1 Loss: {sl1.item():.4f}")
    
    # Test SSD Loss
    ssd_loss = create_ssd_loss()
    
    cls_preds = torch.randn(4, 1000, 17)
    reg_preds = torch.randn(4, 1000, 4)
    cls_targets = torch.randint(0, 17, (4, 1000))
    reg_targets = torch.randn(4, 1000, 4)
    pos_mask = torch.rand(4, 1000) > 0.9  # 10% positifs
    
    total, cls_l, reg_l = ssd_loss(
        cls_preds, reg_preds, cls_targets, reg_targets, pos_mask
    )
    print(f"   SSD Total Loss: {total.item():.4f}")
    print(f"   SSD Cls Loss: {cls_l.item():.4f}")
    print(f"   SSD Reg Loss: {reg_l.item():.4f}")
    
    print("✅ Test réussi!")
