import wandb
from omegaconf import OmegaConf
import hydra

from legged_gym.env import Env
from legged_gym.rl import OnPolicyRunner
from legged_gym.utils.helpers import setSeed, createLogDir

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(config_path="../configs/", config_name="defaults", version_base=None)
def train(cfg):
    # print(OmegaConf.to_yaml(cfg))
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, config=config)

    setSeed(cfg.experiment.seed)
    logdir = createLogDir(cfg)

    # TODO: allow resuming from a specific checkpoint
    # TODO: play script
    # TODO: export script
    # TODO: performance
    # TODO: simplify code (take out optional stuff)
    # TODO: simplify settings

    # TODO change entity in wandb to be loaded from external file

    env = Env(cfg)
    ppo = OnPolicyRunner(env, cfg.ppo, log_dir=logdir, device=cfg.experiment.device, wandb=wandb)
    ppo.learn(num_learning_iterations=cfg.ppo.runner.max_iterations, init_at_random_ep_len=True)

    # model = Model(**wandb.config.model.configs)


if __name__ == "__main__":
    train()
