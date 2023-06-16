from omegaconf import DictConfig, OmegaConf
import hydra
import os

from config import ContrastiveExpConfig


@hydra.main(version_base=None, config_path='../', config_name='config')
def main(cfg: ContrastiveExpConfig):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    print(hydra_cfg['runtime']['output_dir'])
    print(OmegaConf.to_yaml(cfg))
    print(os.getcwd())


if __name__ == '__main__':
    main()
