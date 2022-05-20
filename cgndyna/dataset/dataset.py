from __future__ import annotations

import numpy as np

from dataclasses import dataclass, field

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from cgndyna.util.networks.base import Network
    from cgndyna.config.base import Config


@dataclass
class Dataset:
    nw: Network
    config: Config
    data: np.ndarray = field(init=False)

    def __post_init__(self):
        """Populates `data` attribute with a np.zeros array of size
        (samples size, nodes size, lag size)
        """
        n_samples: int = self.config.exp.num_samples
        n_nodes: int = self.config.nw.num_nodes
        lag: int = self.config.exp.lag
        self.data = np.zeros((n_samples, n_nodes, lag))
