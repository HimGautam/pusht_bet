#!/usr/bin/env python
"""
Behavior Transformer (BeT) policy for PushT image dataset.
Implements k-way bin + k-way offset residual heads with
focal classification loss and masked MSE regression per the paper's MT-Loss.
"""
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from lerobot.constants import OBS_IMAGE, ACTION
from lerobot.policies.pretrained import PreTrainedPolicy
from .configuration_bet import BehaviorTransformerConfig
from collections import deque
from typing import Dict, Tuple, Optional, List, Union

from lerobot.policies.normalize import Normalize, Unnormalize

from lerobot.policies.bet.nanoGPT.model import GPTConfig, Block, LayerNorm


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    alpha: float
) -> torch.Tensor:
    """
    Multi-class focal loss for better handling of class imbalance.
    
    Args:
        logits: (B, K) raw scores for each class
        targets: (B,) integer class labels in range [0, K)
        gamma: focusing parameter (γ≥0). Higher values focus more on hard examples.
        alpha: weighting factor (α∈[0,1]) for addressing class imbalance
    
    Returns:
        Scalar focal loss value averaged over batch
    """
    probs = F.log_softmax(logits, dim=-1)                                       # (B, K)
    log_pt = probs[torch.arange(probs.size(0), device=probs.device), targets]   # (B,)
    pt = log_pt.exp()                                                           # (B,)
    loss = -alpha * (1 - pt).pow(gamma) * log_pt                                # (B,)
    return loss.mean()                                                          # scalar


class SmallCNN(nn.Module):
    def __init__(self, out_dim: int, input_channels: int = 1, input_size: Tuple[int, int] = (96, 96)):
        """
        A small CNN backbone for processing images.
        
        Args:
            out_dim: Output embedding dimension
            input_channels: Number of input image channels (1 for grayscale, 3 for RGB)
            input_size: Input image dimensions (height, width)
        """
        super().__init__()
        self.input_channels = input_channels
        self.input_size = input_size
        
        # Calculate output feature dimensions after convolutions
        h, w = input_size
        h_out = h // 8  # After 3 stride-2 convolutions
        w_out = w // 8
        
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * h_out * w_out, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process image through CNN backbone."""
        # Input validation
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B,C,H,W), got shape {x.shape}")
        if x.shape[1] != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, got {x.shape[1]}")
        
        return self.net(x)


class BehaviorTransformerPolicy(PreTrainedPolicy):
    config_class = BehaviorTransformerConfig
    name = "behavior_transformer"

    def __init__(self, 
                 config: BehaviorTransformerConfig,
                 *,
                 dataset_stats: Dict[str, Dict[str, Tensor]] | None = None,
                 **kwargs):
        """
        Behavior Transformer (BeT) policy implementation.
        
        Args:
            config: Configuration object with BeT parameters
            dataset_stats: Statistics for input/output normalization
            **kwargs: Additional arguments
        """
        super().__init__(config)
        
        torch.backends.cudnn.benchmark = True
        
        # Set device once and use it consistently
        self.device = torch.device(getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu"))
        
        self.in_ch = getattr(config, "image_channels", 3)  # 1 for grayscale, 3 for RGB
        self.context_length = config.context_length

        # Initialize image encoder based on channels
        if self.in_ch == 3:
            # 1) Frozen ResNet-18 backbone → proj to D
            backbone = getattr(models, config.vision_backbone)(
                weights=config.pretrained_backbone_weights
            )
            modules = list(backbone.children())[:-1]  # drop last layers
            self.cnn = nn.Sequential(*modules)        # outputs (B, F, 1, 1)
            print("Using Restnet Backbone with frozne weights")
            for p in self.cnn.parameters():
                p.requires_grad = config.backbone_grad

            feat_dim = backbone.fc.in_features
            self.proj = nn.Linear(feat_dim, config.dim_model)
        else:
            # 1) For grayscale image use Small CNN backbone → proj to D
            input_size = getattr(config, "image_size", (96, 96))
            self.cnn = SmallCNN(out_dim=config.dim_model, 
                               input_channels=self.in_ch,
                               input_size=input_size)
            self.proj = nn.Identity()

        # Project state (2D positions) to model dimension
        self.state_proj = nn.Linear(2, config.dim_model)
        
        # Fusion projection for combining image and state embeddings
        self.fusion_proj = nn.Linear(config.dim_model * 2, config.dim_model)

        # positional embeddings
        self.use_pos_emb = config.use_pos_emb
        if self.use_pos_emb:
            self.pos_emb = nn.Embedding(config.context_length, config.dim_model)

        # 2) Transformer encoder
        gpt_cfg = GPTConfig(
            block_size = config.context_length,
            vocab_size = config.num_bins,   
            n_layer    = config.n_decoder_layers,
            n_head     = config.n_heads,
            n_embd     = config.dim_model,
            dropout    = config.dropout_rate,
            bias       = True,
        )

        self.blocks = nn.ModuleList([
            Block(gpt_cfg) for _ in range(config.n_decoder_layers)
        ])

        self.ln_f = LayerNorm(config.dim_model, bias=gpt_cfg.bias)
        self.token_dropout = nn.Dropout(config.dropout_rate)

        # 3) Bin logits heads + k-way offset heads per dimension
        self.num_bins = config.num_bins
        self.bin_head = nn.Linear(config.dim_model, self.num_bins) 
        
        self.offset_heads = nn.ModuleList([
            nn.Linear(config.dim_model, self.num_bins) for _ in range(2)
        ])

        # Configure normalization with dataset statistics
        with torch.no_grad():
            self.normalize_inputs = Normalize(norm_map=config.normalization_mapping, 
                                            features=config.input_features, 
                                            stats=dataset_stats)
            self.unnormalize_outputs = Unnormalize(norm_map=config.normalization_mapping, 
                                                features=config.output_features, 
                                                stats=dataset_stats)
            self.normalize_targets = Normalize(norm_map=config.normalization_mapping, 
                                            features=config.output_features, 
                                            stats=dataset_stats)

            # 4) Load and normalize k-means centroids (shape: K×2)
            try:
                raw_centroids = torch.load(config.bin_centroids_path).float()
                centroids_kx2 = raw_centroids.T
                # Normalize Centroids
                centroids = self.normalize_targets({"action": centroids_kx2})["action"]
                centroids = centroids.T
                self.register_buffer("centroids", centroids)
            except (FileNotFoundError, RuntimeError) as e:
                raise RuntimeError(f"Failed to load centroids from {config.bin_centroids_path}: {str(e)}")

        # Tracking state
        self.step_counter = 0
        self.frame_buffer = deque(maxlen=config.context_length)
        self.state_buffer = deque(maxlen=config.context_length)

        self.reset()

        # Move model to device
        self.to(self.device)
        
    def _grayscale(self, img: torch.Tensor) -> torch.Tensor:
        """
        Convert RGB image to grayscale.
        
        Args:
            img: Input RGB image tensor of shape (B, 3, H, W)
            
        Returns:
            Grayscale image tensor of shape (B, 1, H, W)
        """
        if img.shape[1] != 3:
            raise ValueError(f"Expected 3-channel RGB input, got {img.shape[1]} channels")
            
        r, g, b = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray.unsqueeze(1)  # Return (B, 1, H, W)
    
    def _prepare_img(self, img: torch.Tensor) -> torch.Tensor:
        """
        Prepare image for processing, handling grayscale conversion if needed.
        
        Args:
            img: Input image tensor of shape (B, C, H, W)
            
        Returns:
            Prepared image tensor with appropriate channel dimension
        """
        if img.ndim != 4:
            raise ValueError(f"Expected 4D input (B,C,H,W), got shape {img.shape}")
        if img.shape[1] not in (1, 3):
            raise ValueError(f"Expected 1 or 3 channels, got {img.shape[1]}")
        
        # Convert RGB to grayscale if needed
        if self.in_ch == 1 and img.shape[1] == 3:
            img = self._grayscale(img)

        return img
    
    def _encode(self, img: torch.Tensor) -> torch.Tensor:
        """
        Encode single frame to transformer token.
        
        Args:
            img: Input image tensor of shape (B, C, H, W)
            
        Returns:
            Image token of shape (1, B, D)
        """
        B = img.size(0)
        feat = self.cnn(img).view(B, -1)      # (B, F)
        tok = self.proj(feat)                 # (B, D)
        return tok.unsqueeze(0)               # (1, B, D)

    def _apply_positional_embedding(self, token: torch.Tensor) -> torch.Tensor:
        """
        Apply positional embedding to token sequence.
        
        Args:
            token: Input token sequence of shape (T, B, D)
            
        Returns:
            Token sequence with positional embeddings
        """
        if self.use_pos_emb:
            T = token.size(0)
            t_idx = torch.arange(T, device=self.device)
            token = token + self.pos_emb(t_idx).unsqueeze(1)
        return token

    def _predict(self, token: torch.Tensor):
        """
        Predict bin logits and offset residual logits.
        
        Args:
            token: Input token sequence of shape (T, B, D)
            
        Returns:
            bin_logits: Classification logits of shape (T, B, K)
            offset_logits: Offset regression logits of shape (T, B, 2, K)
        """
        # Permute to GPT expected format (B, T, D)
        T = token.size(0)
        token_bt = token.permute(1, 0, 2)  # now (B, T, D)

        # Process through transformer blocks
        x = token_bt
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)  # (B, T, D)
        
        # Permute back to (T, B, D)
        outs = x.permute(1, 0, 2)

        # Generate bin and offset logits for each timestep
        bin_logits = torch.stack([self.bin_head(outs[t]) for t in range(T)], dim=0)
        offset_logits = torch.stack([
            torch.stack([h(outs[t]) for h in self.offset_heads], dim=1)  # (B, 2, K)
            for t in range(T)
        ], dim=0)  # (T, B, 2, K)

        return bin_logits, offset_logits

    def forward(self, batch: Dict[str, torch.Tensor]):
        """
        Forward pass for training.
        
        Args:
            batch: Dictionary containing:
                - OBS_IMAGE: Image observations of shape (B, T, C, H, W)
                - ACTION: Ground truth actions of shape (B, T, 2)
                
        Returns:
            loss: Combined loss value
            logs: Dictionary of metrics and diagnostics
        """
        cfg = self.config
        self.step_counter += 1

        # Normalize inputs and targets
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        norm_actions = batch[ACTION]   # (B, T, 2)
        
        # Encode observations into tokens
        seq = batch.get(OBS_IMAGE, None)
        if seq is None:
            raise ValueError("Missing observation image in batch")
            
        if seq.ndim == 5:  # Multi-frame sequence (B, T, C, H, W)
            B, T, C, H, W = seq.shape
            imgs = seq.view(B * T, C, H, W)
            imgs = self._prepare_img(imgs)  # ensure correct channels
            feats = self.cnn(imgs).view(B * T, -1)
            img_token = self.proj(feats).view(B, T, -1)  # (B, T, D)
            
            img_token = img_token.permute(1, 0, 2)  #Converting to (T, B, D)
        else:  # Single frame (B, C, H, W)
            img = self._prepare_img(seq)  
            img_token = self._encode(img)  # (1, B, D)
            T = 1
            B = img_token.size(1)

        # Process state information
        states = batch["observation.state"]
        states_flat = states.view(B * T, 2)
        state_token = self.state_proj(states_flat).view(B, T, -1)
        state_token = state_token.permute(1, 0, 2)   #Converting to (T, B, D)

        # Fuse image and state tokens
        concat_token = torch.cat([img_token, state_token], dim=-1)
        token = self.fusion_proj(concat_token)
        
        # Apply positional embeddings and dropout
        token = self._apply_positional_embedding(token)
        token = self.token_dropout(token)

        # Get predictions
        bin_logits, offset_logits = self._predict(token)

        # Compute true bins by nearest centroids
        diff = norm_actions.unsqueeze(-1) - self.centroids.unsqueeze(0).unsqueeze(0)
        d2 = diff.pow(2).sum(dim=2)
        true_bins = d2.argmin(dim=-1)  # (B, T)

        # === Classification loss ===
        logits_flat = bin_logits.reshape(-1, cfg.num_bins)
        bins_flat = true_bins.permute(1, 0).reshape(-1)
        cls_loss = focal_loss(logits_flat, bins_flat,
                           gamma=cfg.focal_gamma, alpha=cfg.focal_alpha)

        # === Regression loss ===
        # Extract centroids for true bins
        centroids_exp = self.centroids.unsqueeze(0).unsqueeze(0).expand(T, B, 2, -1)  # (T, B, 2, K)
        bin_idx_tb = true_bins.permute(1, 0).unsqueeze(-1)  # (T, B, 1)
        idx_exp = bin_idx_tb.unsqueeze(2).expand(T, B, 2, 1)  # (T, B, 2, 1)
        
        # Get true centroids and residuals
        true_cent = centroids_exp.gather(-1, idx_exp).squeeze(-1)  # (T, B, 2)
        pred_off = offset_logits.gather(-1, idx_exp).squeeze(-1)   # (T, B, 2)
        true_res = norm_actions.permute(1, 0, 2) - true_cent       # (T, B, 2)
        
        # Compute squared error and take mean

        se_x = (pred_off[:, :, 0] - true_res[:, :, 0]).pow(2) 
        se_y = (pred_off[:, :, 1] - true_res[:, :, 1]).pow(2)

        mse_per_t_dim = (se_x.mean(dim=1) + se_y.mean(dim=1)) / 2
        reg_loss = mse_per_t_dim.sum() / T

        # Compute offset MSE metric
        with torch.no_grad():
            offset_mse = mse_per_t_dim.mean()

        # Combine loss terms
        cls_loss_term = cfg.lambda_bin * cls_loss
        reg_loss_term = cfg.lambda_offset * reg_loss
        loss = cls_loss_term + reg_loss_term

        # Compute additional metrics
        with torch.no_grad():
            preds = logits_flat.argmax(dim=-1)
            cls_acc = (preds == bins_flat).float().mean()
            hist = torch.bincount(preds, minlength=self.num_bins)
            num_bins_used = (hist > 0).sum().item()
            top_bin = int(torch.argmax(hist).item())
            top_bin_count = int(hist[top_bin].item())

        logs = {
            "cls_loss_term": cls_loss_term.item(),
            "reg_loss_term": reg_loss_term.item(),
            "cls_acc": cls_acc.item(),
            "offset_mse": offset_mse.item(),
            "num_bins_used": num_bins_used,
            "top_bin": top_bin,
            "top_bin_count": top_bin_count,
        }
        return loss, logs

    def get_optim_params(self):
        """
        Get parameters that require optimization.
        
        Returns:
            List of parameters with requires_grad=True
        """
        return [p for p in self.parameters() if p.requires_grad]

    @torch.no_grad()
    def predict_action_chunk(self, observations: Dict[str, torch.Tensor]):
        """
        Predict actions for a batch of observations.
        
        Args:
            observations: Dictionary containing:
                - OBS_IMAGE: Image observations of shape (B, T, C, H, W) or (B, C, H, W)
                - observation.state: Optional state information
                
        Returns:
            Predicted actions of shape (B, 2)
        """
        self.eval()
        # Normalize inputs
        observations = self.normalize_inputs(observations)

        # Process image observations
        seq = observations.get(OBS_IMAGE)
        if seq is None:
            raise ValueError("Missing observation image")
     
        if seq.ndim == 5:  # Multi-step: (B, T, C, H, W)
            B, T, C, H, W = seq.shape
            imgs = seq.reshape(B * T, C, H, W)
            imgs = self._prepare_img(imgs)
            feats = self.cnn(imgs).view(B * T, -1)
            token = self.proj(feats).view(B, T, -1)  # (T, B, D)
            token = token.permute(1, 0, 2)
        else:  # Single frame: (B, C, H, W)
            img = self._prepare_img(seq)
            token = self._encode(img)  # (1, B, D)
            T = 1
            B = token.size(1)

        # Process state information if present
        st = observations.get("observation.state")
        if st is not None:
            if st.ndim == 2:  # (B, 2)
                st = st.unsqueeze(1)  # (B, 1, 2)
            
            st_flat = st.reshape(-1, 2)
            st_tok = self.state_proj(st_flat).view(B, st.shape[1],  -1)  # (B, T, D)
            st_tok = st_tok.permute(1, 0, 2) #(T, B, D)
            token = torch.cat([token, st_tok], dim=-1)  # (T, B, 2D)
            token = self.fusion_proj(token)  # (T, B, D)
            
            # Apply positional embeddings consistently with training
            token = self._apply_positional_embedding(token)

        # Run through transformer to get predictions
        bin_logits, offset_logits = self._predict(token)
        
        # Get predictions from last timestep
        last_bin = bin_logits[-1]  # (B, K)
        last_offset = offset_logits[-1]  # (B, 2, K)
        
        # Select highest probability bins
        bins = last_bin.argmax(dim=-1)  # (B,)
        
        # Get offsets and centroids for selected bins
        res = last_offset.permute(0, 2, 1).gather(1, bins.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)  # (B, 2)
        cent = self.centroids[:, bins].T  # (B, 2)
        
        # Combine centroid + residual to get final action
        act = cent + res  # (B, 2)
        act = act.unsqueeze(0)  # (1, B, 2)
        
        # Unnormalize to original action space
        out = self.unnormalize_outputs({"action": act})["action"]
        return out[0]  # (B, 2)

    @torch.no_grad()
    def select_action(self, observations: Dict[str, Union[np.ndarray, torch.Tensor]]):
        """
        Select action for a single observation during inference.
        
        Args:
            observations: Dictionary containing:
                - OBS_IMAGE: Single image observation
                - observation.state: Optional state information
                
        Returns:
            Predicted action of shape (1, 2)
            
        """
        self.eval()
        
        # Process image
        img = observations[OBS_IMAGE]
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
            
        if img.ndim == 3:  # (H, W, C) or (C, H, W)
            # Handle different input formats
            if img.shape[2] in (1, 3):  # (H, W, C)
                img = img.permute(2, 0, 1)
            img = img.unsqueeze(0)  # Add batch dimension: (1, C, H, W)


        B = img.shape[0]
        # Add current frame to buffer
        self.frame_buffer.append(img)

        # Stack frames into a sequence
        seq = torch.stack(list(self.frame_buffer), dim=1)  # (1, T, C, H, W)
        obs_batch = {OBS_IMAGE: seq}

        # Process state if present
        if "observation.state" in observations:
            st = observations["observation.state"]
            if isinstance(st, np.ndarray):
                st = torch.from_numpy(st).float()
            if st.ndim == 1:
                st = st.unsqueeze(0)  # (1, 2)
            
            # Add to state buffer
            self.state_buffer.append(st)
            
            # Stack states
            st_seq = torch.stack(list(self.state_buffer), dim=1)  # (1, T, 2)
            obs_batch["observation.state"] = st_seq

      
        if len(self.frame_buffer) < self.context_length:
            #If we dont have observations of context length, inject random actions
            last_pos = st
            max_off = 5.0
            # Sample random bin index
            delta = (torch.rand(B, 2, device=self.device) - 0.5) * 2 * max_off
            random_action = (last_pos + delta).clamp(0.0, 512.0)
            # Squeeze to (1,2) so it matches predict_action_chunk output
            return random_action

        
          # Get prediction
        action = self.predict_action_chunk(obs_batch)
        return action

    def reset(self):
        """Reset internal buffers when starting a new episode."""
        self.frame_buffer.clear()
        self.state_buffer.clear()