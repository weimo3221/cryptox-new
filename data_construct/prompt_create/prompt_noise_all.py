noise_template = """Answer the following question. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question (For multiple-choice questions, please provide only the answer options).

Some words in the question may be subject to the following noise injection rules:
{noise_rule}
Furthermore, these words will also be transformed according to the dictionary provided below:
- dictionary:
{rule}

Think step by step before answering.

### Question:

{question}

{options}"""

noise_template_math = """Answer the following question. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question.

Some words in the question may be subject to the following noise injection rules:
{noise_rule}
Furthermore, these words will also be transformed according to the dictionary provided below:
- dictionary:
{rule}

Think step by step before answering.

### Question:

{question}"""


noise_template_mbpp = """Write the Python code according to the following question. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question.

Some words in the question may be subject to the following noise injection rules:
{noise_rule}
Furthermore, these words will also be transformed according to the dictionary provided below:
- dictionary:
{rule}

Think step by step before answering.

### Question:

{question}

Your code should pass these tests: \n{test_list}
There may be libraries you need to import: \n{test_imports}"""
