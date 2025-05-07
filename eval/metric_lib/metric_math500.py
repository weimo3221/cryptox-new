# eval/metric_lib/metric_math500.py
# Python Standard Library
import re
import asyncio
import logging
import os

# Commonly Used Open-Source Libraries
from openai import OpenAI

# Libraries for projects
from .prompt_template_for_metric import judge_prompt_template_math500

# Setting up OpenAI API keys and API libraries to use vLLM's API server
openai_api_key = "API-KEY"
openai_api_base = "https://api.deepseek.com"
model_name = "deepseek-chat"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# Configuring logging, setting the logging level and output format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def normalize_response(response: str) -> str:
    """
    Normalize responses by removing markdown and LaTeX formatting that may prevent matches
    """
    return (
        response.replace("**", "")
        .replace("$\\boxed{", "")
        .replace("}$", "")
        .replace("\\$", "")
        .replace("$\\text{", "")
        .replace("$", "")
        .replace("\\mathrm{", "")
        .replace("\\{", "")
        .replace("\\text", "")
        .replace("\\(", "")
        .replace("\\mathbf{", "")
        .replace("{", "")
        .replace("\\boxed", "")
    )


def judge_correctness_math500(problem, real_answer, generated_answer):
    """
    Determine the correctness of the answers to the MATH dataset
    """
    generated_answer = normalize_response(generated_answer)
    pos = generated_answer.lower().rfind("answer")
    if pos == -1:
        return False
    generated_answer = generated_answer[pos:]
    ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*(.*)"
    match_for_generated_answer = re.findall(ANSWER_PATTERN_MULTICHOICE, generated_answer)
    if match_for_generated_answer:
        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": judge_prompt_template_math500.format(problem=problem, real_answer=real_answer, generated_answer=match_for_generated_answer[-1])},
                ]
            )
        except:
            return False
        match = re.findall(r"<answer>(.*?)</answer>", normalize_response(chat_response.choices[-1].message.content))
        if match:
            return match[-1] == "correct"
        else:
            logging.error(f"The <answer> tag was not found in the response. problem:{problem}\nreal_answer:{real_answer}\ngenerated_answer:{generated_answer}")
            return False  # Or processed on demand
    else:
        return False


async def judge_correctness(problem, real_answer, generated_answer):
    """
    Asynchronous version of the request function for determining correctness, using a running event loop
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, judge_correctness_math500, problem, real_answer, generated_answer)
    return result


if __name__ == '__main__':
    problem="The binary number $10101001110_{2}$ is equal to what number in base eight?"
    real_answer="2516_8"
    generated_answer="123124Answer:2516"
    print(judge_correctness_math500(problem,real_answer,generated_answer))