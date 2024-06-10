import wandb
from omegaconf import OmegaConf
import hydra

from legged_gym.env import Env
from legged_gym.rl import OnPolicyRunner

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(config_path="../configs/", config_name="defaults", version_base=None)
def train(cfg):
    print(OmegaConf.to_yaml(cfg))
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # TODO: set seed from cfg.ppo.runner or cfg.env
    # TODO: do log dir automatically if none specified. And resume if specified
    # TODO: wandb logging
    # TODO: reduce model write freq
    # TODO: play script
    # TODO: export script
    # TODO: simplify code (take out optional stuff)
    # TODO: simplify settings


    env = Env(cfg)
    ppo = OnPolicyRunner(env, cfg.ppo, log_dir="./test", device=cfg.env.device)
    ppo.learn(num_learning_iterations=cfg.ppo.runner.max_iterations, init_at_random_ep_len=True)

    # env, env_cfg = task_registry.make_env(name=args.task, args=args)
    # ppo_runner, train_cfg = tasksk_registry.make_alg_runner(env=env, name=args.task, args=args)
    # logger.log_params(Args = vars(train_cfg))
    # ppo_runner.learn(num_learning_epochsg_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    # wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
    # wandb.log({"loss": loss})
    # model = Model(**wandb.config.model.configs)


if __name__ == "__main__":
    train()
