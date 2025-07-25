from dataclasses import dataclass, field
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode, FeatureType
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

import torch
import os
from lerobot.constants import OBS_IMAGE, ACTION # Make sure these are imported if not already
from typing import List, Optional


@PreTrainedConfig.register_subclass("behavior_transformer")
@dataclass
class BehaviorTransformerConfig(PreTrainedConfig):
    """
    configuration class for the Behavior Transformer policy.
    """
    
    # Context/chunking parameters
    context_length: int = 4  # Number of frames to consider in the context
    n_action_steps: int = 4  # Number of action steps to predict
    chunk_size: int = 4  # equal to context_length for now



    # ─── Tell the dataset to return windows of length `n_obs_steps` ────
    sample_traj_windows: bool = True
    window_size:         int  = 4

     


    # Input Normalization
    normalization_mapping: dict[FeatureType, NormalizationMode] = field(
        default_factory=lambda: {
            FeatureType.VISUAL: NormalizationMode.MEAN_STD,
            FeatureType.STATE: NormalizationMode.MEAN_STD,
            FeatureType.ACTION: NormalizationMode.MEAN_STD,  # Min-Max normalization for actions
        }
    )
      # request both image and state from the dataset
    input_features: list[str] = field(
        default_factory=lambda: [OBS_IMAGE, "observation.state"]
    )
    # specify that we only predict the 'action' key
    output_features: list[str] = field(
        default_factory=lambda: [ACTION]
    )

    

    # Transformer Architecture
    dim_model: int = 256
    n_heads: int = 4
    dim_feedforward: int = 512
    n_decoder_layers: int = 5
    pre_norm: bool = True
    feedforward_activation: str = "gelu"
    
    use_pos_emb : bool = True  # Use positional embeddings in the transformer
   


    # Action Binning & clustering (K means)
    num_bins: int = 32
    action_max: float = 512.0
    print("Current working directory:", os.getcwd())
    bin_centroids_path: str = field(default_factory=lambda: "pushT_bin_centroids.pt")


    # Vision Backbone
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: bool = False
    backbone_grad: bool = False

    image_channels: int = 1 #to choose Restnet18 , 1 chooses custom cnn with grayscale images

    # Training hyperparameters
    optimizer_lr: float = 1.0e-4
    optimizer_weight_decay: float = 0.034 #0.1
    optimizer_lr_backbone: float = 0.0 # No backbone training by default
    lambda_offset: float = 12   # Weight for the offset loss
    lambda_bin: float = 1.0 # Weight for cls loss
    gradient_clip_norm: float = 1.0  # 1.0 Gradient clipping norm
    dropout_rate: float = 0.35 #0.1
    
    
    # Focal Loss parameters
    focal_gamma: float = 2.0
    focal_alpha: float = .25

    # ─── Scheduler parameters ──────────────────────────────────
    scheduler_warmup_steps: int = 3_000
    scheduler_decay_steps: int = 20_000
    scheduler_decay_lr: float = 5e-5



    

    def __post_init__(self):
        super().__post_init__()
        self.chunk_size = self.context_length  
        self.window_size = self.context_length 
        self.n_action_steps = self.context_length 

        
    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr,
                           weight_decay=self.optimizer_weight_decay,
                           grad_clip_norm=self.gradient_clip_norm)
    

    def validate_features(self) -> None:
        if not self.input_features:
            raise ValueError("Behavior Transformer requires input features to be defined.")
        
    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )


    @property
    def observation_delta_indices(self) -> dict[str, list[int]]:
        return list(range(-(self.context_length - 1), 1)) # e.g., for 4 steps: [-3, -2, -1, 0]
        
    @property
    def reward_delta_indices(self) -> None:
        return None
    
    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(-(self.context_length - 1), 1))