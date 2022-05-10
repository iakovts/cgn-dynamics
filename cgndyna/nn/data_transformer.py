from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cgndyna.config.base import Config
    from cgndyna.util.networks.base import Network


class DataTransformer:
    def __init__(self, config: Config, network: Network):
        self.cfg = config
        self.nw = network
        
