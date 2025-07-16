EMPTY_PLACEHOLDER = "(Empty)"

INPUT_HEADER = "<INPUT_BLOCK>"
INPUT_SEPARATOR = "<INPUT_SEPARATOR>"
INPUT_FOOTER = "</INPUT_BLOCK>"

TOOL_HEADER = "<TOOL_BLOCK>"
TOOL_FOOTER = "</TOOL_BLOCK>"

INPUT_WITH_NOTE = """
# Input from a previous agent
<input>
{input}
</input>
# Note from a previous agent
<note>
{note}
</note>
""".strip()

NOTE_NO_ACTION = """
The previous agent did not terminate by calling the "terminate" function. Instead, it returned the following output:
<output>
{output}
</output>
This might be what the previous agent wanted to leave for you.
""".strip()

OUTPUT_NOTE_INFO = """
# Output
<output>
{output}
</output>
# Note
<note>
{note}
</note>
""".strip()

TOOL_INFO = """
# Tool Name
<tool_name>
{tool_name}
</tool_name>
# Tool Parameters
<tool_parameters>
{tool_parameters}
</tool_parameters>
"""

TOOL_RESULT_INFO = """
# Tool Result
<tool_result>
{tool_result}
</tool_result>
""".strip()

DONT_CHANGE_HEADER = "<DONT_CHANGE>"
DONT_CHANGE_FOOTER = "</DONT_CHANGE>"
