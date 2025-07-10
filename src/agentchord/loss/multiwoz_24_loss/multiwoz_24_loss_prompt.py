JOINT_GOAL_ACCURACY_LOSS = """
The system has made the following user intention predictions:
{prediction}
The ground truth user intentions are:
{ground_truth}
The false positive predictions are:
{false_positive}
The false negative predictions are:
{false_negative}
""".strip()

INFORM_SUCCESS_LOSS = """
The system has made the following queries:
{provided_queries}
The ground truth queries are:
{requested_queries}
The system has provided the following information:
{provided_information}
The ground truth information is:
{requested_information}
""".strip()