from ..metadata import BaseMetaData


class BaseLoss:
    """
    Base class for all loss functions.
    """

    def __init__(self):
        pass

    def compute_loss(self, prediction: BaseMetaData, evaluation_result: BaseMetaData) -> str:
        """
        Compute the loss given predictions and targets.
        
        Args:
            prediction (BaseMetaData): The metadata containing the predictions.
            evaluation_result (BaseMetaData): The metadata containing the evaluation results.
        
        Returns:
            The computed loss value.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def __repr__(self):
        return "BaseLoss"