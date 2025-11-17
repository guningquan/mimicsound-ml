from mimicsound.configs.config import Config
from mimicsound.configs.base_config import config_factory, get_all_registered_configs

# note: these imports are needed to register these classes in the global config registry
from mimicsound.configs.mimicplay_config import MimicPlayConfig
from mimicsound.configs.act_config import ACTConfig, MimicSoundConfig