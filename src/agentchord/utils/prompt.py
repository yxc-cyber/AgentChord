EMPTY_PLACEHOLDER = "(Empty)"

INPUT_WITH_NOTE = """
# Input from the previous agent
{input}
# Note from the previous agent
{note}
""".strip()

NOTE_NO_ACTION = """
The previous agent did not terminate by calling the "terminate" function. Instead, it returned the following output:
{output}
This might be what the previous agent wanted to leave for you.
""".strip()