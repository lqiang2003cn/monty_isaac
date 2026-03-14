"""Supervised pretraining experiment for CNN-style multi-patch Monty.

The upstream ``MontySupervisedObjectPretrainingExperiment`` assumes one
physical sensor position per learning module (``self.sensor_pos[i]`` for
each LM ``i``).  In our CNN-style setup we have 100 virtual patches from
a single physical camera, so ``sensor_pos`` only has 1 entry.

This subclass expands ``sensor_pos`` to match the number of LMs — all
set to the same camera position [0, 0, 0].  Since all patches share the
same viewpoint, the multi-LM offset (``lm_offset``) is zero everywhere,
which is geometrically correct.
"""

from __future__ import annotations

import numpy as np

from tbp.monty.frameworks.experiments.pretraining_experiments import (
    MontySupervisedObjectPretrainingExperiment,
)


class CNNStylePretrainingExperiment(MontySupervisedObjectPretrainingExperiment):
    """Pretraining variant that handles N LMs backed by a single physical sensor."""

    def setup_experiment(self, config):
        super().setup_experiment(config)
        n_lms = len(self.model.learning_modules)
        if self.sensor_pos.shape[0] < n_lms:
            self.sensor_pos = np.tile(self.sensor_pos[0:1], (n_lms, 1))
