# eval/metric_lib/metric_bbh.py
# Python Standard Library
import re
import asyncio
import logging
import os

# Commonly Used Open-Source Libraries
from openai import OpenAI

# Libraries for projects
from .prompt_template_for_metric import judge_prompt_template_needle

# Setting up OpenAI API keys and API libraries to use vLLM's API server
openai_api_key = "API-KEY"
openai_api_base = "https://api.deepseek.com"
model_name = "deepseek-chat"
logging.basicConfig(level=logging.ERROR)
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


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


def judge_correctness_needle(problem, real_answer, generated_answer):
    """
    Determining the correctness of answers in the needle dataset
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
                    {"role": "user", "content": judge_prompt_template_needle.format(problem=problem, answer=real_answer,response=match_for_generated_answer[-1])},
                ]
            )
        except Exception as e:
            print(e)
            return False
        match = re.findall(r"<answer>(.*?)</answer>", chat_response.choices[-1].message.content)
        # print(chat_response.choices[-1].message.content)
        # match = re.findall(r'(?i)"answer_score"\s*:\s*(.*)', chat_response.choices[-1].message.content)
        if match:
            try:
                score = int(match[-1].replace("\n", "").replace(" ", "")) / 3
                return score
            except Exception as e:
                # Wrong answer
                return 0
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
    result = await loop.run_in_executor(None, judge_correctness_needle, problem, real_answer, generated_answer)
    return result

if __name__ == '__main__':
    problem="girls, boys"
    real_answer="girls, boys"
    generated_answer="123124Answer: boys"
    print(judge_correctness_needle(problem,real_answer,generated_answer))