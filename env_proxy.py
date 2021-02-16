"""TODO: Create an 'environment proxy' that relays observations / actions etc from a remote environment via gRPC.

For now this simply holds the 'remote' environment in memory.
"""
from sequoia.settings import Environment


class EnvironmentProxy(Environment):
    def __init__(self):
        pass
