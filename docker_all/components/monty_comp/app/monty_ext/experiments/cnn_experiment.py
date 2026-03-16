"""Supervised pretraining experiment for CNN-style multi-patch Monty.

The upstream ``MontySupervisedObjectPretrainingExperiment`` assumes one
physical sensor position per learning module (``self.sensor_pos[i]`` for
each LM ``i``).  In our CNN-style setup we have 100 virtual patches from
a single physical camera, so ``sensor_pos`` only has 1 entry.

This subclass:
  1. Overrides ``__init__`` to bypass ``OmegaConf.to_object()`` which rejects
     plain-dict configs containing Python class references.
  2. Expands ``sensor_pos`` to match the number of LMs — all set to the same
     camera position [0, 0, 0].  Since all patches share the same viewpoint,
     the multi-LM offset (``lm_offset``) is zero everywhere, which is
     geometrically correct.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tbp.monty.frameworks.experiments.monty_experiment import MontyExperiment
from tbp.monty.frameworks.experiments.pretraining_experiments import (
    MontySupervisedObjectPretrainingExperiment,
)


class CNNStylePretrainingExperiment(MontySupervisedObjectPretrainingExperiment):
    """Pretraining variant that handles N LMs backed by a single physical sensor."""

    def __init__(self, config):
        # MontySupervisedObjectPretrainingExperiment.__init__ calls
        # OmegaConf.to_object(config) which rejects plain dicts that contain
        # Python class references.  Replicate parent logic without OmegaConf.
        output_dir = Path(config["logging"]["output_dir"])
        config["logging"]["output_dir"] = output_dir / "pretrained"
        self.first_epoch_object_location = {}
        MontyExperiment.__init__(self, config)

    def setup_experiment(self, config):
        super().setup_experiment(config)
        n_lms = len(self.model.learning_modules)
        if self.sensor_pos.shape[0] < n_lms:
            self.sensor_pos = np.tile(self.sensor_pos[0:1], (n_lms, 1))
