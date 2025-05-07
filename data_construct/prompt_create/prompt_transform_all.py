transform_template = """Answer the following question. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question (For multiple-choice questions, please provide only the answer options).

Some words in the question will be transformed according to the following rules:
1.Duplicate each string. For example, 'happy' becomes 'hhaappppyy'
2.Shift each letter one position to the right in the alphabet. For example, 'happy' becomes 'ibqqz'
3.Shift a string one position to the right, cyclically. For example, 'happy' becomes 'yhapp'
4.Reverse a string. For example, 'happy' becomes 'yppah'
5.Shift a string's characters two positions to the left. For example, 'happy' becomes 'ppyha'
6.Rotate the letters in even positions. For example, 'happy' becomes 'hbpqy'
7.Rotate the letters in odd positions. For example, 'happy' becomes 'iaqpz'
8.Convert letters into their corresponding emoji strings based on the dictionary provided below.
- dictionary: 
{rule}

Think step by step before answering.

### Question:

{question}

{options}"""

transform_template_math = """Answer the following question. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question.

Some words in the question will be transformed according to the following rules:
1.Duplicate each string. For example, 'happy' becomes 'hhaappppyy'
2.Shift each letter one position to the right in the alphabet. For example, 'happy' becomes 'ibqqz'
3.Shift a string one position to the right, cyclically. For example, 'happy' becomes 'yhapp'
4.Reverse a string. For example, 'happy' becomes 'yppah'
5.Shift a string's characters two positions to the left. For example, 'happy' becomes 'ppyha'
6.Rotate the letters in even positions. For example, 'happy' becomes 'hbpqy'
7.Rotate the letters in odd positions. For example, 'happy' becomes 'iaqpz'
8.Convert letters into their corresponding emoji strings based on the dictionary provided below.
- dictionary: 
{rule}

Think step by step before answering.

### Question:

{question}"""


transform_template_mbpp = """Write the Python code according to the following question. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question.

Some words in the question will be transformed according to the following rules:
1.Duplicate each string. For example, 'happy' becomes 'hhaappppyy'
2.Shift each letter one position to the right in the alphabet. For example, 'happy' becomes 'ibqqz'
3.Shift a string one position to the right, cyclically. For example, 'happy' becomes 'yhapp'
4.Reverse a string. For example, 'happy' becomes 'yppah'
5.Shift a string's characters two positions to the left. For example, 'happy' becomes 'ppyha'
6.Rotate the letters in even positions. For example, 'happy' becomes 'hbpqy'
7.Rotate the letters in odd positions. For example, 'happy' becomes 'iaqpz'
8.Convert letters into their corresponding emoji strings based on the dictionary provided below.
- dictionary: 
{rule}

Think step by step before answering.

### Question:

{question}

Your code should pass these tests: \n{test_list}
There may be libraries you need to import: \n{test_imports}"""
