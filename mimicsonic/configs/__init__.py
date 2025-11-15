from mimicsonic.configs.config import Config
from mimicsonic.configs.base_config import config_factory, get_all_registered_configs

# note: these imports are needed to register these classes in the global config registry
from mimicsonic.configs.mimicplay_config import MimicPlayConfig
from mimicsonic.configs.act_config import ACTConfig, MimicSonicConfig