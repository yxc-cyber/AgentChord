import json
from typing import Literal, Optional

from ...gbc_object import GBC, GBCBase
from ...metadata import MultiWOZ24MetaData
from ..base_loss import BaseLoss
from .multiwoz_24_loss_prompt import INFORM_SUCCESS_LOSS, JOINT_GOAL_ACCURACY_LOSS


class MultiWOZ24Loss(BaseLoss):
    """
    Loss function for MultiWOZ 2.4 dataset.
    This class computes the loss based on the predictions and targets from the MultiWOZ 2.4 dataset.
    """

    def __init__(self):
        super().__init__()

    def compute_loss(
            self,
            prediction: Optional[MultiWOZ24MetaData] = None,
            evaluation_result: Optional[MultiWOZ24MetaData] = None,
            type: Literal["joint_goal_accuracy", "inform_success"] = "joint_goal_accuracy"
        ) -> str:
        """
        Compute the loss given predictions and targets.
        
        Args:
            meta_data (BaseMetaData): The metadata containing the predictions and targets.
        
        Returns:
            The computed loss value.
        """
        if type == "inform_success":
            return self._compute_inform_success_loss(evaluation_result)
        elif type == "joint_goal_accuracy":
            return self._compute_joint_goal_accuracy_loss(prediction, evaluation_result)
        else:
            raise ValueError(f"Unknown loss type: {type}. Supported types are 'joint_goal_accuracy' and 'inform_success'.")

    def _compute_joint_goal_accuracy_loss(
            self,
            prediction: MultiWOZ24MetaData,
            evaluation_result: MultiWOZ24MetaData
        ) -> str:
        """
        Compute the joint goal accuracy loss.
        """
        system_response = prediction.system_response
        groundtruth_dialogue_state = prediction.groundtruth_dialogue_state
        dialogue_state = evaluation_result.dialogue_state
        joint_goal_accuracy_detail = evaluation_result.joint_goal_accuracy_detail
        loss = JOINT_GOAL_ACCURACY_LOSS.format(
            prediction=json.dumps(dialogue_state),
            ground_truth=json.dumps(groundtruth_dialogue_state),
            false_positive=json.dumps(joint_goal_accuracy_detail.get("false_positive", {})),
            false_negative=json.dumps(joint_goal_accuracy_detail.get("false_negative", []))
        )
        if isinstance(system_response, GBCBase):
            loss = GBC(loss, connections=system_response.get_connections(), weights=[1.0])
        else:
            loss = GBC(loss, connections=system_response, weights=1.0)
        return loss

    def _compute_inform_success_loss(
            self,
            evaluation_result: MultiWOZ24MetaData
        ) -> str:
        """
        Compute the inform success loss.
        """
        system_response = evaluation_result.system_response
        inform = evaluation_result.inform
        inform_detail = evaluation_result.inform_detail
        success = evaluation_result.success
        success_detail = evaluation_result.success_detail

        provided_queries = dict()
        requested_queries = dict()
        for domain in inform:
            if domain != "total" and inform[domain] == 0.0:
                provided_queries[domain] = inform_detail.get(domain, {}).get("provided", [])
                requested_queries[domain] = inform_detail.get(domain, {}).get("requested", [])
        provided_information = dict()
        requested_information = dict()
        for domain in success:
            if domain != "total" and success[domain] == 0.0:
                provided_information[domain] = list(success_detail.get(domain, {}).get("provided", set()))
                requested_information[domain] = list(success_detail.get(domain, {}).get("requested", set()))
        loss = INFORM_SUCCESS_LOSS.format(
            provided_queries=json.dumps(provided_queries),
            requested_queries=json.dumps(requested_queries),
            provided_information=json.dumps(provided_information),
            requested_information=json.dumps(requested_information)
        )

        connections = list()
        if isinstance(system_response, list):
            for response in system_response:
                if isinstance(response, GBCBase):
                    connections.extend(response.get_connections())
                else:
                    connections.append(response)
        else:
            connections = system_response.get_connections()
        loss = GBC(
            loss,
            connections=connections,
            weights=[1.0]* len(connections)
        )
        return loss