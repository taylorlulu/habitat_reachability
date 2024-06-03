import habitat
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.nav import NavigationTask
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config as get_baselines_config

if __name__ == "__main__":
    config = get_baselines_config(
        "/home/lu/Desktop/embodied_ai/habitat-challenge/habitat-lab/habitat_baselines/config/pointnav/ppo_pointnav_example.yaml"
    )

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    trainer = trainer_init(config)
    trainer.train()