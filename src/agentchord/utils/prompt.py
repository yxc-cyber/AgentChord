EMPTY_PLACEHOLDER = "(Empty)"

INPUT_HEADER = "<INPUT_BLOCK>"
INPUT_SEPARATOR = "<INPUT_SEPARATOR>"
INPUT_FOOTER = "</INPUT_BLOCK>"

TOOL_HEADER = "<TOOL_BLOCK>"
TOOL_FOOTER = "</TOOL_BLOCK>"

INPUT = """
# Input from a previous agent
<content>
{content}
</content>
""".strip()

# OUTPUT_NO_ACTION = """
# The agent did not terminate by calling the "terminate" function. Instead, it returned the following content:
# {content}
# This might be what the previous agent wanted to leave for you.
# """.strip()

OUTPUT_NO_ACTION = """
{content}
""".strip()

OUTPUT_TOO_MANY_ACTIONS = """
The agent terminated by calling the "terminate" function because it performed too many actions and exceeded the maximum number of actions allowed.
""".strip()

OUTPUT_TOO_MUCH_THINKING = """
The agent terminated by calling the "terminate" function because it thought too much and exceeded the maximum number of thought tokens allowed.
""".strip()

OUTPUT_INFO = """
# Output
<content>
{output}
</content>
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
Some inputs may be useless, so you only focus on the useful inputs.
When you decide to generate an output for the next agent or the user, you should use the "terminate" tool to return the output.
If you are instructed to output a certain format, put the output in the "output" field of the "terminate" tool.
If you are instructed to output a JSON object, please put the JSON object in the "output" field of the "terminate" tool. Don't take the keys of the JSON object as the parameters of a tool.
Please make the response/output as concise as possible. Try to limit it within 128 tokens.
Don't generate a response/output without putting it in the "terminate" tool.
""".strip()

USER_PROMPT_TEMPLATE = """
{prompt}

**Important Note**
You are a user simulator.
""".strip()