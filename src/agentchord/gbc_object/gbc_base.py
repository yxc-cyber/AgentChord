import random
from copy import copy
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from ..agent_system import BaseAgent


class GBCBase:
    def bind_connections(self, connections: Union[list, Any], weights: Union[List[float], float]) -> None:
        if not isinstance(connections, list):
            connections = [connections]
        if not isinstance(weights, list):
            weights = [weights]
        if connections and not weights:
            weights = [1.0] * len(connections)
        assert len(connections) == len(weights),\
            f"Connections and weights must have the same length. Self: {self}, Connections: {connections}, Weights: {weights}"
            
        self.connections = connections
        self.weights = weights

    def add_connection(self, connections: Union[list, Any], weights: Union[List[float], float]) -> None:
        if not isinstance(connections, list):
            connections = [connections]
        if not isinstance(weights, list):
            weights = [weights]
        assert len(connections) == len(weights), "Connections and weights must have the same length."
        self.connections.extend(connections)
        self.weights.extend(weights)

    def get_connections(self) -> list:
        return self.connections
    
    def get_weights(self) -> list:
        return self.weights
    
    def bind_subject(self, subject: Union[str, "BaseAgent"]) -> None:
        self.subject = subject

    def get_subject(self) -> Optional[Union[str, "BaseAgent"]]:
        return self.subject
    
    def get_node_representation(self) -> str:
        """
        Returns a string representation of the node.
        This method should be overridden by subclasses to provide specific representations.
        """
        return f"GBC Object ({self.subject if self.subject else 'Unnamed Subject'}): \n{self}"
    
    def backward(self, bandwidth: int = 3, cache: Optional[List[List[Tuple[Union[str, "BaseAgent"], Any]]]] = None) -> None:
        """
        Backward pass for the GBC object.
        """
        from ..agent_system import BaseAgent, Input
        from ..loss import BaseLoss

        # Update the cache with the current subject and self.
        if cache is None:
            cache = [[(self.subject, self)]]
        else:
            new_cache = list()
            for trajectory in cache:
                trajectory = copy(trajectory)
                trajectory.insert(0, (self.subject, self))
                new_cache.append(trajectory)
            cache = new_cache
        # If the subject is a BaseAgent, append the optimization info to it.
        if self.subject and (isinstance(self.subject, BaseAgent) or isinstance(self.subject, Input)):
            self.subject.append_optimization_info(cache)
        # Select the strongest connections based on the weights. Break ties in random order.
        if (self.subject and isinstance(self.subject, BaseLoss)) \
            or all(weight  == 1.0 for weight in self.weights) \
            or not self.connections:
            selected_connections = self.connections
        else:
            connections_and_weights = list(zip(self.connections, self.weights))
            shuffled_connections_and_weights = random.sample(connections_and_weights, len(connections_and_weights))
            sorted_indices = sorted(
                range(len(shuffled_connections_and_weights)),
                key=lambda i: shuffled_connections_and_weights[i][1],
                reverse=True
            )[:bandwidth]
            selected_connections = [shuffled_connections_and_weights[i][0] for i in sorted_indices]
        for connection in selected_connections:
            if isinstance(connection, GBCBase):
                connection.backward(bandwidth=bandwidth, cache=cache)