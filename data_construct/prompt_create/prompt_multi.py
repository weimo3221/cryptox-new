multi_template = """Answer the following question. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question (For multiple-choice questions, please provide only the answer options).

Some words are transformed sequentially according to the following rules:
{multi_rule}

Think step by step before answering.

### Question:

{question}

{options}"""

multi_template_math = """Answer the following question. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question.

Some words are transformed sequentially according to the following rules:
{multi_rule}

Think step by step before answering.

### Question:

{question}"""


multi_template_mbpp = """Write the Python code according to the following question. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question.

Some words are transformed sequentially according to the following rules:
{multi_rule}

Think step by step before answering.

### Question:

{question}

Your code should pass these tests: \n{test_list}
There may be libraries you need to import: \n{test_imports}"""


multi_rule = {"noise": "- Add random letters to the odd-numbered positions within the string. For example, 'happy' becomes 'hiapjpyk'\n",
              "des": "- The string is encoded into bytes using UTF-8.\n- Next, these bytes are padded to a length of 8 bytes using PKCS7 padding before being encrypted with DES, resulting in encrypted bytes.\n- The encrypted bytes will be directly converted into a hexadecimal string representation.\n- The DES key is: {key}\n",
              "transform": """- Duplicate each string. For example, 'happy' becomes 'hhaappppyy'
- Shift each letter one position to the right in the alphabet. For example, 'happy' becomes 'ibqqz'
- Shift a string one position to the right, cyclically. For example, 'happy' becomes 'yhapp'
- Reverse a string. For example, 'happy' becomes 'yppah'
- Shift a string's characters two positions to the left. For example, 'happy' becomes 'ppyha'
- Rotate the letters in even positions. For example, 'happy' becomes 'hbpqy'
- Rotate the letters in odd positions. For example, 'happy' becomes 'iaqpz'
- Convert letters into their corresponding emoji strings based on the dictionary provided below.
- dictionary: 
{rule}"""
}