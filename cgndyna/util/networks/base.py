from abc import abstractmethod, ABC


class Network(ABC):
    def __init__(self, config):
        self.cfg = config

    @abstractmethod
    def network_generator(self):
        ...
