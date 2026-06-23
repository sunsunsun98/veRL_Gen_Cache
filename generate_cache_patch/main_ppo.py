"""Thin generate-cache entrypoint built on top of verl.trainer.main_ppo."""

import hydra
import ray

from generate_cache_patch.ray_trainer import GenerateCacheRayPPOTrainer
from verl.trainer import main_ppo as ppo_main


class GenerateCacheTaskRunner(ppo_main.TaskRunner):
    """Use the upstream PPO task runner with a generate-cache trainer class."""

    def run(self, config):
        ppo_main.RayPPOTrainer = GenerateCacheRayPPOTrainer
        return super().run(config)


@hydra.main(config_path="../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    """PPO main entrypoint that injects GenerateCacheRayPPOTrainer."""
    ppo_main.auto_set_device(config)
    config = ppo_main.migrate_legacy_reward_impl(config)
    task_runner_class = ray.remote(num_cpus=1)(GenerateCacheTaskRunner)
    ppo_main.run_ppo(config, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
