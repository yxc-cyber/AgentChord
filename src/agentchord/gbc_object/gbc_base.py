from typing import Any, List, Optional, Union


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
    
    def bind_subject(self, subject: str) -> None:
        self.subject = subject

    def get_subject(self) -> Optional[str]:
        return self.subject
    
    def get_node_representation(self) -> str:
        """
        Returns a string representation of the node.
        This method should be overridden by subclasses to provide specific representations.
        """
        return f"GBC Object ({self.subject if self.subject else 'Unnamed Subject'}): \n{self}"