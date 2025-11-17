"""
Implementation of mimicsound.
"""

from collections import OrderedDict
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import robomimic.utils.tensor_utils as TensorUtils
from mimicsound.algo import register_algo_factory_func, PolicyAlgo
from robomimic.algo.bc import BC_VAE

from mimicsound.utils.mimicsoundUtils import nds
import matplotlib.pyplot as plt
import robomimic.utils.obs_utils as ObsUtils

from mimicsound.configs import config_factory

from mimicsound.models.act_nets import Transformer, StyleEncoder
from mimicsound.models.audio_nets import create_ast_encoder
from mimicsound.utils.audio_utils import generate_audio_data_for_batch, get_audio_feature_dim

from robomimic.models.transformers import PositionalEncoding

import robomimic.models.base_nets as BaseNets
import mimicsound.models.policy_nets as PolicyNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils

import json

from mimicsound.algo.act import ACT, ACTModel


class BidirectionalCrossModalAttention(nn.Module):
    """
    Bidirectional cross-modal attention between audio and visual features.
    """
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1, modality_name="unknown"):
        super().__init__()
        self.modality_name = modality_name
        
        # Audio → Visual attention
        self.audio_to_visual = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        # Visual → Audio attention  
        self.visual_to_audio = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer normalization
        self.norm_audio = nn.LayerNorm(hidden_dim)
        self.norm_visual = nn.LayerNorm(hidden_dim)
        
        # Feed-forward networks
        self.ffn_audio = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.ffn_visual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.norm_audio_ffn = nn.LayerNorm(hidden_dim)
        self.norm_visual_ffn = nn.LayerNorm(hidden_dim)
    
    def forward(self, audio_feat, visual_feat):
        """
        Args:
            audio_feat: [B, 1, hidden_dim] - Audio features
            visual_feat: [B, S, hidden_dim] - Visual features (S = H*W*num_cameras)
        
        Returns:
            audio_enhanced: [B, 1, hidden_dim] - Enhanced audio features
            visual_enhanced: [B, S, hidden_dim] - Enhanced visual features
            audio_attn_weights: [B, 1, S] - Audio attention weights
            visual_attn_weights: [B, S, 1] - Visual attention weights
        """
        # First layer: Cross-modal attention
        # Audio attends to visual features
        audio_attended, audio_attn_weights = self.audio_to_visual(
            audio_feat, visual_feat, visual_feat
        )
        audio_feat = self.norm_audio(audio_feat + audio_attended)
        
        # Visual attends to audio features
        visual_attended, visual_attn_weights = self.visual_to_audio(
            visual_feat, audio_feat, audio_feat
        )
        visual_feat = self.norm_visual(visual_feat + visual_attended)
        
        # Second layer: Feed-forward networks
        audio_ffn = self.ffn_audio(audio_feat)
        audio_feat = self.norm_audio_ffn(audio_feat + audio_ffn)
        
        visual_ffn = self.ffn_visual(visual_feat)
        visual_feat = self.norm_visual_ffn(visual_feat + visual_ffn)
        
        return audio_feat, visual_feat, audio_attn_weights, visual_attn_weights


@register_algo_factory_func("mimicsound")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    algo_class, algo_kwargs = MimicSound, {}
    return algo_class, algo_kwargs


class MimicSoundModel(ACTModel):
    def __init__(
            self,
            backbones,
            transformer,
            encoder,
            latent_dim,
            a_dim,
            state_dim,
            num_queries,
            camera_names,
            num_channels,
            audio_enabled=False,
            audio_config=None,
            cross_modal_attention_config=None,
    ):
        super().__init__(
            backbones,
            transformer,
            encoder,
            latent_dim,
            a_dim,
            state_dim,
            num_queries,
            camera_names,
            num_channels
        )

        hidden_dim = transformer.d
        if a_dim == 7:
            hand_state_dim = 3
            hand_action_dim = 3
        elif a_dim == 14:
            hand_state_dim = 6
            hand_action_dim = 6
        
        # Audio modal support
        self.audio_enabled = audio_enabled
        if self.audio_enabled:
            # Create shared AST encoder for both robot and human data
            audio_config = audio_config or {}
            self.audio_encoder = create_ast_encoder(**audio_config)
            self.audio_feature_dim = get_audio_feature_dim()
            
            # Project audio features to hidden dimension
            self.audio_proj = nn.Linear(self.audio_feature_dim, hidden_dim)
            
            # Reinitialize additional_pos_embed to support 3 inputs (latent, proprio, audio)
            self.additional_pos_embed = nn.Embedding(3, hidden_dim)  # @gnq important !!
        else:
            self.audio_encoder = None
            self.audio_feature_dim = 0
            self.audio_proj = None
        
        # Dual bidirectional cross-modal attention configuration
        self.cross_modal_attention_config = cross_modal_attention_config or {}
        if self.cross_modal_attention_config.get('enabled', False) and self.audio_enabled:
            # Create two independent bidirectional attention mechanisms
            self.audio_robot_attention = BidirectionalCrossModalAttention(
                hidden_dim=hidden_dim,
                num_heads=self.cross_modal_attention_config.get('num_heads', 8),
                dropout=self.cross_modal_attention_config.get('dropout', 0.1),
                modality_name="audio-robot"
            )
            self.audio_human_attention = BidirectionalCrossModalAttention(
                hidden_dim=hidden_dim,
                num_heads=self.cross_modal_attention_config.get('num_heads', 8),
                dropout=self.cross_modal_attention_config.get('dropout', 0.1),
                modality_name="audio-human"
            )
            print(f"✅ Dual bidirectional cross-modal attention enabled:")
            print(f"   - Audio-Robot attention: {self.audio_robot_attention.modality_name}")
            print(f"   - Audio-Human attention: {self.audio_human_attention.modality_name}")
            print(f"   - Config: {self.cross_modal_attention_config}")
        else:
            self.audio_robot_attention = None
            self.audio_human_attention = None
            if self.cross_modal_attention_config.get('enabled', False) and not self.audio_enabled:
                print("⚠️  Cross-modal attention requested but audio is disabled. Disabling cross-modal attention.")
        
        self.robot_transformer_input_proj = nn.Linear(state_dim, hidden_dim)
        self.robot_action_head = nn.Linear(hidden_dim, a_dim)

        self.encoder_action_proj = nn.Linear(
            a_dim, hidden_dim
        )  # project robot action to embedding
        self.encoder_joint_proj = nn.Linear(
            state_dim, hidden_dim
        )  # project robot qpos to embedding

        self.hand_transformer_input_proj = nn.Linear(hand_state_dim, hidden_dim)
        self.hand_encoder_action_proj = nn.Linear(
            hand_action_dim, hidden_dim
        )  # project action to embedding
        self.hand_encoder_joint_proj = nn.Linear(
            hand_state_dim, hidden_dim
        )  # project qpos to embedding
        self.hand_action_head = nn.Linear(hidden_dim, hand_action_dim)

    def forward(self, qpos, image, env_state, modality, actions, is_pad=None, audio=None):
        if modality == "robot":
            return self._forward(
                qpos,
                actions,
                image,
                self.encoder_action_proj,
                self.encoder_joint_proj,
                self.robot_transformer_input_proj,
                self.robot_action_head,
                camera_names=self.camera_names,
                is_pad=is_pad,
                aux_action_head=self.hand_action_head,
                audio=audio,
                modality=modality,
            )
        elif modality == "hand":
            assert "front_img" in self.camera_names[0], "hand modality assumes first camera is front_img"
            # print("hand pos shape is", qpos.shape)
            # exit()
            return self._forward(
                qpos,
                actions,
                image,
                self.hand_encoder_action_proj,
                self.hand_encoder_joint_proj,
                self.hand_transformer_input_proj,
                self.hand_action_head,
                camera_names=self.camera_names[:1],
                is_pad=is_pad,
                audio=audio,
                modality=modality,
            )

    def _forward(self, qpos, actions, image, encoder_action_proj=None, encoder_joint_proj=None, transformer_input_proj=None, action_head=None, camera_names=None, env_state=None, is_pad=None, aux_action_head=None, audio=None, modality=None):
        """
        Custom forward method for MimicSoundModel with audio support.
        """
        is_training = actions is not None
        batch_size = qpos.size(0)

        if is_training:
            actions_encod = encoder_action_proj(actions)
            qpos_encod = encoder_joint_proj(qpos)
            # Use StyleEncoder to get latent distribution and sample
            dist = self.encoder(qpos_encod, actions_encod)
            mu = dist.mean
            logvar = dist.scale.log() * 2
            latent_sample = dist.rsample()
        else:
            # Inference mode, use zeros for latent vector
            mu = logvar = None
            latent_sample = torch.zeros(batch_size, self.latent_dim, device=qpos.device)

        latent_input = self.latent_out_proj(latent_sample)  # [batch_size, hidden_dim]

        all_cam_features = []

        for cam_id in range(len(camera_names)):
            features = self.backbones[cam_id](image[:, cam_id])
            features = self.input_proj(features)
            all_cam_features.append(features)

        src = torch.cat(all_cam_features, dim=-1)  # [B, hidden_dim, H, W * num_cameras]

        batch_size, hidden_dim, height, width = src.shape
        src = src.flatten(2).permute(0, 2, 1)  # [B, S, hidden_dim], S = H * W * num_cameras]

        proprio_input = transformer_input_proj(qpos).unsqueeze(1)  # [B, 1, hidden_dim]
        latent_input = latent_input.unsqueeze(1)  # [B, 1, hidden_dim]
        # input("press enter to continue")
        # Debug: Check audio configuration
        # print("audio_enabled is:", self.audio_enabled)
        # print("audio is not None is:", audio is not None)
        # if audio is not None:
        #     print("audio shape:", audio.shape)
        # Process audio if enabled
        audio_input = None
        if self.audio_enabled and audio is not None:
            # audio shape: [batch_size, audio_samples] - raw audio for the entire sequence
            # Process the entire audio sequence at once
            audio_feat = self.audio_encoder(audio)  # [batch_size, audio_feature_dim]
            
            # Project to hidden dimension
            audio_input = self.audio_proj(audio_feat).unsqueeze(1)  # [batch_size, 1, hidden_dim]
            # exit()

        # Dual bidirectional cross-modal attention between audio and visual features
        if audio_input is not None:
            # Determine which attention mechanism to use based on modality
            if modality == "robot" and self.audio_robot_attention is not None:
                # print(f"🎯 Applying Audio-Robot bidirectional cross-modal attention...")
                # print(f"   Audio input shape: {audio_input.shape}")
                # print(f"   Robot visual features shape: {src.shape}")
                # print(f"   Number of robot cameras: {len(camera_names)}")
                
                # Apply audio-robot bidirectional cross-modal attention
                audio_enhanced, visual_enhanced, audio_attn_weights, visual_attn_weights = self.audio_robot_attention(
                    audio_input, src
                )
                
                # print(f"   Enhanced audio shape: {audio_enhanced.shape}")
                # print(f"   Enhanced robot visual shape: {visual_enhanced.shape}")
                # print(f"   Audio attention weights shape: {audio_attn_weights.shape}")
                # print(f"   Robot visual attention weights shape: {visual_attn_weights.shape}")
                
                # Use enhanced features
                audio_input = audio_enhanced
                src = visual_enhanced
                
            elif modality == "hand" and self.audio_human_attention is not None:
                # print(f"🎯 Applying Audio-Human bidirectional cross-modal attention...")
                # print(f"   Audio input shape: {audio_input.shape}")
                # print(f"   Human visual features shape: {src.shape}")
                # print(f"   Number of human cameras: {len(camera_names)}")
                
                # Apply audio-human bidirectional cross-modal attention
                audio_enhanced, visual_enhanced, audio_attn_weights, visual_attn_weights = self.audio_human_attention(
                    audio_input, src
                )
                
                # print(f"   Enhanced audio shape: {audio_enhanced.shape}")
                # print(f"   Enhanced human visual shape: {visual_enhanced.shape}")
                # print(f"   Audio attention weights shape: {audio_attn_weights.shape}")
                # print(f"   Human visual attention weights shape: {visual_attn_weights.shape}")
                
                # Use enhanced features
                audio_input = audio_enhanced
                src = visual_enhanced

        query_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, num_queries, hidden_dim]

        tgt = query_embed # tgt = torch.zeros_like(query_embed) + query_embed. ACT passes zeros to decoder
        
        # Concatenate inputs
        src_inputs = [latent_input, proprio_input]
        if audio_input is not None:
            src_inputs.append(audio_input)
        src_inputs.append(src)
        src = torch.cat(src_inputs, axis=1)  # [B, S + 2/3, hidden_dim]


        # print("latent_input shape is:", latent_input.shape)
        # print("proprio_input shape is:", proprio_input.shape)
        # print("audio_input shape is:", audio_input.shape)
        # print("src shape is:", src.shape)
        # exit()
        
        # Learnable additional pos embed for latent input, proprio input, and optionally audio input
        num_additional_inputs = 2 if audio_input is None else 3
        additional_pos_embed = self.additional_pos_embed.weight[:num_additional_inputs].unsqueeze(0).repeat(batch_size, 1, 1) #[B, 2/3, hidden_dim]
        src[:, :num_additional_inputs, :] += additional_pos_embed 

        hs_queries = self.transformer(src, tgt) # [B, tgt, hidden_dim]

        action_pred = action_head(hs_queries)  # [B, num_queries, action_dim]
        is_pad_pred = self.is_pad_head(hs_queries)  # [B, num_queries, 1]

        # aux action head for 2 head output
        if aux_action_head:
            action_pred_aux = aux_action_head(hs_queries)
            return (action_pred, action_pred_aux), is_pad_pred, [mu, logvar]

        return action_pred, is_pad_pred, [mu, logvar]


class MimicSound(ACT):
    def build_model_opt(self, policy_config):
        backbones = []
        if len(policy_config["camera_names"]) > 0:
            for cam_name in policy_config["camera_names"]:
                backbone_class_name = policy_config["backbone_class_name"]
                backbone_kwargs = policy_config["backbone_kwargs"]

                try:
                    backbone_class = getattr(BaseNets, backbone_class_name)
                except AttributeError:
                    raise ValueError(f"Unsupported backbone class: {backbone_class_name}")
                
                backbone = backbone_class(**backbone_kwargs)
                backbones.append(backbone)
        else:
            backbones = None

        if backbones is not None:
            # assume camera input shape is same for all TODO dynamic size
            cam_name = policy_config["camera_names"][0]  
            input_shape = self.obs_key_shapes[cam_name]  # (C, H, W)
            num_channels = backbones[0].output_shape(input_shape)[0]
        else:
            num_channels = None

        transformer = Transformer(
            d=policy_config["hidden_dim"],
            h=policy_config["nheads"],
            d_ff=policy_config["dim_feedforward"],
            num_layers=policy_config["dec_layers"],
            dropout=policy_config["dropout"],
        )

        style_encoder = StyleEncoder(
            act_len=policy_config["action_length"],
            hidden_dim=policy_config["hidden_dim"],
            latent_dim=policy_config["latent_dim"],
            h=policy_config["nheads"],
            d_ff=policy_config["dim_feedforward"],
            num_layers=policy_config["enc_layers"],
            dropout=policy_config["dropout"],
        )

        # Audio configuration
        audio_enabled = policy_config.get("audio_enabled", False)
        audio_config = policy_config.get("audio_config", {})
        
        model = MimicSoundModel(
            backbones=backbones,
            transformer=transformer,
            encoder=style_encoder,
            latent_dim=policy_config["latent_dim"],
            a_dim=policy_config["a_dim"],
            state_dim=policy_config["state_dim"],
            num_queries=policy_config["num_queries"],
            camera_names=policy_config["camera_names"],
            num_channels=num_channels,
            audio_enabled=audio_enabled,
            audio_config=audio_config,
            cross_modal_attention_config=policy_config["cross_modal_attention_config"],
        )

        model.cuda()

        return model

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.nets = nn.ModuleDict()
        self.chunk_size = self.global_config["train"]["seq_length"]
        self.camera_keys = self.obs_config["modalities"]["obs"]["rgb"].copy()
        self.proprio_keys = self.obs_config["modalities"]["obs"]["low_dim"].copy()
        self.obs_keys = self.proprio_keys + self.camera_keys

        self.proprio_dim = 0
        for k in self.proprio_keys:
            self.proprio_dim += self.obs_key_shapes[k][0]

        backbone_class_name = self.obs_config["encoder"]["rgb"]["core_kwargs"][
            "backbone_class"
        ]
        backbone_kwargs = self.obs_config["encoder"]["rgb"]["core_kwargs"][
            "backbone_kwargs"
        ]

        # Audio configuration - use dictionary access to get updated values
        print(f"=== _create_networks debug ===")
        print(f"self.global_config type: {type(self.global_config)}")
        print(f"self.global_config keys: {list(self.global_config.keys())}")
        print(f"audio_enabled key exists: {'audio_enabled' in self.global_config}")
        print(f"audio_config key exists: {'audio_config' in self.global_config}")
        
        # Debug: Check what's actually in the config
        print(f"Direct access self.global_config['audio_enabled']: {self.global_config['audio_enabled']}")
        print(f"Direct access self.global_config['audio_config']: {self.global_config['audio_config']}")
        
        # Use dictionary access to get the updated values from config.update()
        audio_enabled = self.global_config.get("audio_enabled", False)
        
        # Get audio config from observation encoder config to avoid duplication
        if hasattr(self.global_config, 'observation') and hasattr(self.global_config.observation, 'encoder') and hasattr(self.global_config.observation.encoder, 'audio'):
            audio_config = self.global_config.observation.encoder.audio.core_kwargs
            print("Using audio config from observation.encoder.audio.core_kwargs")
        else:
            # Fallback to separate audio_config if observation config not available
            audio_config = self.global_config.get("audio_config", {})
            print("Using fallback audio_config")
        
        print(f"audio_enabled (dict access): {audio_enabled}")
        print(f"audio_config (dict access): {audio_config}")

        # Get cross-modal attention configuration
        cross_modal_attention_config = self.global_config.get("cross_modal_attention", {})
        print(f"cross_modal_attention_config: {cross_modal_attention_config}")

        policy_config = {
            "num_queries": self.global_config.train.seq_length,
            "hidden_dim": self.algo_config.act.hidden_dim,
            "dim_feedforward": self.algo_config.act.dim_feedforward,
            "backbone": self.algo_config.act.backbone,
            "enc_layers": self.algo_config.act.enc_layers,
            "dec_layers": self.algo_config.act.dec_layers,
            "nheads": self.algo_config.act.nheads,
            "latent_dim": self.algo_config.act.latent_dim,
            "action_length": self.chunk_size,
            "a_dim": self.ac_dim,
            "ac_key": self.ac_key,
            "state_dim": self.proprio_dim,
            "camera_names": self.camera_keys,
            "backbone_class_name": backbone_class_name,
            "backbone_kwargs": backbone_kwargs,
            "dropout": self.algo_config.act.get("dropout", 0.1),
            "audio_enabled": audio_enabled,
            "audio_config": audio_config,
            "cross_modal_attention_config": cross_modal_attention_config,
        }

        self.kl_weight = self.algo_config.act.kl_weight
        model = self.build_model_opt(policy_config)    

        self.nets["policy"] = model
        self.nets = self.nets.float().to(self.device)

        self.temporal_agg = False
        self.query_frequency = self.chunk_size  # TODO maybe tune

        self._step_counter = 0
        self.a_hat_store = None

        rand_kwargs = self.global_config.observation.encoder.rgb.obs_randomizer_kwargs
        self.color_jitter = transforms.ColorJitter(
            brightness=(rand_kwargs.brightness_min, rand_kwargs.brightness_max), 
            contrast=(rand_kwargs.contrast_min, rand_kwargs.contrast_max), 
            saturation=(rand_kwargs.saturation_min, rand_kwargs.saturation_max), 
            hue=(rand_kwargs.hue_min, rand_kwargs.hue_max)
        )

        # MimicSound specific setup
        self.proprio_keys_hand = (
            self.global_config.observation_hand.modalities.obs.low_dim.copy()
        )

        self.ac_key_hand = self.global_config.train.ac_key_hand
        self.ac_key_robot = self.global_config.train.ac_key
    
    def process_batch_for_training(self, batch, ac_key):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["obs"] = {
            k: batch["obs"][k][:, 0, :]
            for k in batch["obs"]
            if k != "pad_mask" and k != "type"
        }
        input_batch["obs"]["pad_mask"] = batch["obs"]["pad_mask"]
        input_batch["goal_obs"] = batch.get(
            "goal_obs", None
        )  # goals may not be present

        if self.ac_key_hand in batch:
            input_batch[self.ac_key_hand] = batch[self.ac_key_hand]
        if self.ac_key_robot in batch:
            input_batch[self.ac_key_robot] = batch[self.ac_key_robot]

        if "type" in batch:
            input_batch["type"] = batch["type"]

        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def _robomimic_to_act_data(self, batch, cam_keys, proprio_keys):
        qpos, images, env_state, actions, is_pad = super()._robomimic_to_act_data(batch, cam_keys, proprio_keys)
        actions_hand = batch.get(self.ac_key_hand, None)
        actions_robot = batch[self.ac_key_robot] if self.ac_key_robot in batch else None

        # Process audio data if available
        audio = None
        if hasattr(self.nets["policy"], "audio_enabled") and self.nets["policy"].audio_enabled:
            # Check if audio data exists in batch["obs"]
            if "obs" in batch and "audio" in batch["obs"]:
                audio = batch["obs"]["audio"]
                # print(f"Found audio in batch['obs']['audio'] with shape: {audio.shape}")
            else:
                # Generate random audio data for debugging
                batch_size = qpos.size(0)
                seq_length = actions.size(1) if actions is not None else 100  # Default sequence length
                audio = generate_audio_data_for_batch(
                    batch_size=batch_size,
                    seq_length=seq_length,
                    device=qpos.device,
                    debug_mode=True
                )
                print(f"Generated random audio with shape: {audio.shape}")
                # exit()

        return qpos, images, env_state, actions_hand, actions_robot, is_pad, audio

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            predictions (dict): dictionary containing network outputs
        """

        modality = self._modality_check(batch)
        cam_keys = (
            self.camera_keys if modality == "robot" else self.camera_keys[:1]
        )  # TODO Simar rm hardcoding
        proprio_keys = (
            self.proprio_keys_hand if modality == "hand" else self.proprio_keys
        )
        qpos, images, env_state, actions_hand, actions_robot, is_pad, audio = self._robomimic_to_act_data(
            batch, cam_keys, proprio_keys
        )
    
        # img = images[0, 0].detach().float().cpu()  # 取第一条
        # img = (img - img.min()) / (img.max() - img.min() + 1e-6)  # 简单归一化到 [0,1]
        # plt.figure()
        # plt.imshow(img.permute(1, 2, 0))
        # plt.title(f"{modality} cam0")
        # plt.axis("off")
        # plt.show()          # 手动关闭窗口后才进入下一次

        actions = actions_hand if modality == "hand" else actions_robot

        a_hat, is_pad_hat, (mu, logvar) = self.nets["policy"](
            qpos, images, env_state, modality, actions=actions, is_pad=is_pad, audio=audio
        )
        total_kld, dim_wise_kld, mean_kld = self.kl_divergence(mu, logvar)
        loss_dict = dict()

        if modality == "hand":
            all_l1 = F.l1_loss(actions_hand, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean() * self.global_config.algo.sp.hand_lambda
            total_kld = total_kld * self.global_config.algo.sp.hand_lambda
        elif modality == "robot":
            all_l1_robot = F.l1_loss(actions_robot, a_hat[0], reduction="none")
            all_l1_hand = F.l1_loss(actions_hand, a_hat[1], reduction="none")
            l1 = (all_l1_robot * ~is_pad.unsqueeze(-1)).mean() + (all_l1_hand * ~is_pad.unsqueeze(-1)).mean()

        loss_dict["l1"] = l1
        loss_dict["kl"] = total_kld[0]

        predictions = OrderedDict(
            actions=actions,
            kl_loss=loss_dict["kl"],
            reconstruction_loss=loss_dict["l1"],
        )

        return predictions

    def forward_eval(self, batch, unnorm_stats):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            predictions (dict): dictionary containing network outputs
        """

        modality = self._modality_check(batch)

        cam_keys = (
            self.camera_keys if modality == "robot" else self.camera_keys[:1]
        )  # TODO Simar rm hardcoding
        proprio_keys = (
            self.proprio_keys_hand if modality == "hand" else self.proprio_keys
        )
        qpos, images, env_state, _, _, is_pad, audio = self._robomimic_to_act_data(
            batch, cam_keys, proprio_keys
        )
        a_hat, is_pad_hat, (mu, logvar) = self.nets["policy"](
            qpos, images, env_state, modality, actions=None, is_pad=is_pad, audio=audio
        )

        # a_hat = a_hat[0] if modality == "robot" else a_hat
        if modality == "robot":
            predictions = OrderedDict()
            predictions[self.ac_key_robot] = a_hat[0]
            predictions[self.ac_key_hand] = a_hat[1]
            predictions = ObsUtils.unnormalize_batch(predictions, unnorm_stats)
        else:
            predictions = OrderedDict()
            predictions[self.ac_key_hand] = a_hat
            predictions = ObsUtils.unnormalize_batch(predictions, unnorm_stats)

        return predictions

class TestModel:
    def __init__(self, config_path):
        ext_cfg = self.load_config(config_path)
        config = config_factory(ext_cfg["algo_name"])

        with config.values_unlocked():
            config.update(ext_cfg)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ac_dim = 7  # Action dimension
        obs_key_shapes = {
            'joint_positions': (7,),
            'front_img_1': (3, 480, 640),
            'right_wrist_img': (3, 480, 640),
        }

        self.act_algo = ACT(
            algo_config=config.algo,  # Use the appropriate section from the config object
            obs_config=config.observation,
            global_config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=ac_dim,
            device=device,
        )

        self.act_algo.train_config = config.train

        # Create networks
        self.act_algo._create_networks()

    def load_config(self, config_path):
        """Load the config from a JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)

    def run_test(self):
        batch_size = 2
        seq_length = self.act_algo.train_config.seq_length  # Access via the config object

        # Create a dummy batch
        dummy_batch = {
            'obs': {
                'joint_positions': torch.randn(batch_size, seq_length, *self.act_algo.obs_key_shapes['joint_positions']),
                'front_img_1': torch.randint(0, 256, (batch_size, seq_length, *self.act_algo.obs_key_shapes['front_img_1']), dtype=torch.uint8),
                'right_wrist_img': torch.randint(0, 256, (batch_size, seq_length, *self.act_algo.obs_key_shapes['right_wrist_img']), dtype=torch.uint8),
                'pad_mask': torch.ones(batch_size, seq_length, 1),
            },
            'actions_joints_act': torch.randn(batch_size, seq_length, self.act_algo.ac_dim),
        }

        # Process the batch for training
        batch = self.act_algo.process_batch_for_training(dummy_batch, 'actions_joints_act')

        print("Processed Batch:", batch)

        # Move batch to device
        batch = self.to_device(batch, self.act_algo.device)

        # Ensure the model is in training mode
        self.act_algo.nets['policy'].train()

        # Perform forward pass for training
        predictions = self.act_algo._forward_training(batch)

        # Compute losses
        losses = self.act_algo._compute_losses(predictions, batch)

        print("Predictions:")
        for key, value in predictions.items():
            print(f"{key}: {value}")

        print("\nLosses:")
        for key, value in losses.items():
            print(f"{key}: {value.item() if isinstance(value, torch.Tensor) else value}")

    def to_device(self, batch, device):
        """Utility function to move data to the specified device."""
        if isinstance(batch, dict):
            return {k: self.to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self.to_device(v, device) for v in batch]
        elif isinstance(batch, torch.Tensor):
            return batch.to(device)
        else:
            return batch
