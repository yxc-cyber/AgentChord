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

NOTE_TOO_MANY_ACTIONS = """
The previous agent terminated by calling the "terminate" function because it performed too many actions and exceeded the maximum number of actions allowed.
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
""".strip()

TOOL_RESULT_INFO = """
# Tool Result
<tool_result>
{tool_result}
</tool_result>
""".strip()

DONT_CHANGE_HEADER = "<DONT_CHANGE>"
DONT_CHANGE_FOOTER = "</DONT_CHANGE>"

PROMPT_TEMPLATE = f"""
{{prompt}}

**Important Note**
When you receive inputs from previous agents, they will be wrapped in {INPUT_HEADER} and {INPUT_FOOTER} tags and separated by {INPUT_SEPARATOR}.
Some inputs may be attached with a note, which contains additional information beyond the input itself.
Some inputs may be useless, so you only focus on the useful inputs.
""".strip()