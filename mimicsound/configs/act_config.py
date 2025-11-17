"""
Config for BC algorithm.
"""

from mimicsound.configs.base_config import BaseConfig
from mimicsound.configs.config import Config


class ACTConfig(BaseConfig):
    ALGO_NAME = "act"

    def train_config(self):
        """
        BC algorithms don't need "next_obs" from hdf5 - so save on storage and compute by disabling it.
        """
        super(ACTConfig, self).train_config()
        self.train.hdf5_load_next_obs = False

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config`
        argument to the constructor. Any parameter that an algorithm needs to determine its
        training and test-time behavior should be populated here.
        """

        # optimization parameters
        self.algo.optim_params.policy.optimizer_type = "adamw"
        self.algo.optim_params.policy.learning_rate.initial = (
            5e-5  # policy learning rate
        )
        self.algo.optim_params.policy.learning_rate.decay_factor = (
            1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.algo.optim_params.policy.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.algo.optim_params.policy.learning_rate.scheduler_type = (
            "linear"  # learning rate scheduler ("multistep", "linear", etc)
        )
        self.algo.optim_params.policy.regularization.L2 = (
            0.0001  # L2 regularization strength
        )

        # loss weights
        self.algo.loss.l2_weight = 0.0  # L2 loss weight
        self.algo.loss.l1_weight = 1.0  # L1 loss weight
        self.algo.loss.cos_weight = 0.0  # cosine loss weight

        # ACT policy settings
        self.algo.act.hidden_dim = 512  # length of (s, a) seqeunces to feed to transformer - should usually match train.frame_stack
        self.algo.act.dim_feedforward = (
            3200  # dimension for embeddings used by transformer
        )
        self.algo.act.backbone = "resnet18"  # number of transformer blocks to stack
        self.algo.act.enc_layers = 4  # number of attention heads for each transformer block (should divide embed_dim evenly)
        self.algo.act.dec_layers = (
            7  # dropout probability for embedding inputs in transformer
        )
        self.algo.act.nheads = (
            8  # dropout probability for attention outputs for each transformer block
        )
        self.algo.act.latent_dim = 32  # latent dim of VAE
        self.algo.act.kl_weight = 20  # KL weight of VAE

        # Playdata training/inference settings
        self.algo.playdata.enable = (
            False  # whether to train with plan data (unlabeled, no-cut)
        )
        self.algo.playdata.goal_image_range = [
            100,
            200,
        ]  # goal image sampling range during training
        self.algo.playdata.eval_goal_gap = 150  # goal image sampling gap during evaluation rollouts (mid of training goal_image_range)
        self.algo.playdata.do_not_lock_keys()


class MimicSoundConfig(ACTConfig):
    ALGO_NAME = "mimicsound"

    def train_config(self):
        """
        BC algorithms don't need "next_obs" from hdf5 - so save on storage and compute by disabling it.
        """
        super(MimicSoundConfig, self).train_config()
        self.train.ac_key_hand = "actions_xyz"
        self.train.dataset_keys_hand = ["actions_xyz"]
        self.train.seq_length_hand = 1
        self.train.hdf5_2_filter_key = "train"

    def observation_config(self):
        super(MimicSoundConfig, self).observation_config()
        
        # Add audio modality support for main observation
        self.observation.modalities.obs.audio = []
        self.observation.modalities.goal.audio = []
        
        # Add audio encoder configuration
        self.observation.encoder.audio.core_class = "ASTEncoder"
        self.observation.encoder.audio.core_kwargs = Config()
        self.observation.encoder.audio.core_kwargs.do_not_lock_keys()
        self.observation.encoder.audio.obs_randomizer_class = None
        self.observation.encoder.audio.obs_randomizer_kwargs = Config()
        self.observation.encoder.audio.obs_randomizer_kwargs.do_not_lock_keys()
        
        # Configure observation_hand
        self.observation_hand.modalities.obs.low_dim = ["joint_positions"]
        self.observation_hand.modalities.obs.rgb = ["front_img_1"]
        self.observation_hand.modalities.obs.audio = []
    
    def algo_config(self):
        super(MimicSoundConfig, self).algo_config()
        self.algo.sp.hand_lambda = 1.0
    
    def meta_config(self):
        super(MimicSoundConfig, self).meta_config()
        print("=== MimicSoundConfig.meta_config() called ===")
        # Add audio configuration support - always set these keys to ensure they exist
        self.audio_enabled = False  # Default value, will be overridden by JSON config
        print("Set audio_enabled = False")
        self.audio_debug = False    # Default value, will be overridden by JSON config
        print("Set audio_debug = False")
        # Always create audio_config to ensure the key exists
        self.audio_config = Config()
        if hasattr(self.audio_config, 'do_not_lock_keys'):
            self.audio_config.do_not_lock_keys()
        print("Set audio_config = Config()")
        
        # Add cross-modal attention configuration support
        self.cross_modal_attention = Config()
        self.cross_modal_attention.enabled = False  # Default value, will be overridden by JSON config
        self.cross_modal_attention.num_heads = 8
        self.cross_modal_attention.dropout = 0.1
        self.cross_modal_attention.use_residual = True
        self.cross_modal_attention.use_ffn = True
        if hasattr(self.cross_modal_attention, 'do_not_lock_keys'):
            self.cross_modal_attention.do_not_lock_keys()
        print("Set cross_modal_attention = Config()")
        print("=== MimicSoundConfig.meta_config() finished ===")
