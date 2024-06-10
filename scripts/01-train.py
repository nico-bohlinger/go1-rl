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
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)

    setSeed(cfg.seed)
    logdir = createLogDir(cfg)

    # TODO: wandb logging
    # TODO: reduce model write freq
    # TODO: allow resuming from a specific checkpoint
    # TODO: play script
    # TODO: export script
    # TODO: simplify code (take out optional stuff)
    # TODO: simplify settings


    env = Env(cfg)
    ppo = OnPolicyRunner(env, cfg.ppo, log_dir=logdir, device=cfg.env.device)
    ppo.learn(num_learning_iterations=cfg.ppo.runner.max_iterations, init_at_random_ep_len=True)


    # wandb.log({"loss": loss})
    # model = Model(**wandb.config.model.configs)


if __name__ == "__main__":
    train()
