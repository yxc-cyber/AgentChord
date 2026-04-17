import json
from typing import Optional

from ...gbc_object import GBC, GBCBase
from ...metadata import TaubenchMetaData
from ..base_loss import BaseLoss
from .taubench_loss_prompt import TAUBENCH_LOSS


class TauBenchLoss(BaseLoss):
    """
    Loss function for TauBench tasks.
    This class computes the loss based on reward details in the evaluation result.
    """

    def __init__(self):
        super().__init__()

    def compute_loss(
            self,
            evaluation_result: Optional[TaubenchMetaData] = None,
        ) -> str:
        """
        Compute loss from TauBench evaluation metadata.
        """
        if evaluation_result is None:
            raise ValueError("evaluation_result cannot be None for TauBench loss computation.")

        reward_details = getattr(evaluation_result, "reward_details", None) or {}
        if not isinstance(reward_details, dict):
            reward_details = {}

        groundtruth_actions = reward_details.get("groundtruth_actions", [])
        predicted_actions = reward_details.get("actions", [])
        action_match = reward_details.get("action_match", False)
        required_output_strings = reward_details.get("groundtruth_outputs", [])
        predicted_responses = reward_details.get("responses", [])
        output_match = reward_details.get("output_match", False)

        loss = TAUBENCH_LOSS.format(
            groundtruth_actions=json.dumps(groundtruth_actions),
            predicted_actions=json.dumps(predicted_actions),
            action_match=json.dumps(action_match),
            required_output_strings=json.dumps(required_output_strings),
            predicted_responses=json.dumps(predicted_responses),
            output_match=json.dumps(output_match),
        )

        responses = evaluation_result.responses
        connections = []
        if isinstance(responses, list):
            for response in responses:
                if isinstance(response, GBCBase):
                    connections.extend(response.get_connections())
                else:
                    connections.append(response)
        elif isinstance(responses, GBCBase):
            connections = responses.get_connections()
        elif responses:
            connections = [responses]

        loss = GBC(
            loss,
            connections=connections,
            weights=[1.0] * len(connections),
            subject=self,
        )
        return loss

    def __repr__(self):
        return "TauBenchLoss"
