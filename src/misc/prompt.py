import random
import openai
from tqdm import tqdm
import time
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="gpt-3.5-turbo", help="The model to use for text generation")
parser.add_argument("-p", "--prompt", type=str, default='Analyze the sentiment of the following reviews and classify them as either "positive" or "negative"', help="The prompt to use for text generation")
parser.add_argument("-t", "--test_data", type=str, default="data/processed/imdb/test.csv", help="The dataset to be labeled by the LLM")
parser.add_argument("-tr", "--train_data", type=str, default="data/processed/imdb/train.csv", help="The train data to use")

args = parser.parse_args()

data = pd.read_csv(args.train_data)
test = pd.read_csv(args.test_data)
model = args.model
PROMPT = args.prompt

num_retries = 10
initial_backoff = 0.5

def get_prompt(data: pd.DataFrame) -> str:
    """
    Get the previous sentiment from the data.

    Parameters
    ----------
    data
        The data to get the previous sentiment from.

    Returns
    -------
    str
        The previous sentiment.
    """

    text = data["text"].to_list()
    label_name = data["label_name"].to_list()


    data = []
    for i in range(len(text)):
        data.append({"role": "user", "content": f"Review: {text[i]}"})
        data.append({"role": "assistant", "content": f"{label_name[i]}"})
    # shuffle the data
    random.shuffle(data)
    instructions = [{"role": "system", "content": PROMPT}]
    instructions.extend(data)
    return instructions

def get_sentiment(text:str, data: pd.DataFrame, model:str = 'gpt-3.5-turbo', num_retries=10, initial_backoff=0.5) -> str:


    # openai.api_type = "azure"
    # openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    # openai.api_version = "2023-07-01-preview"
    openai.api_key = os.getenv("OPENAI_API_KEY")


    prompt = get_prompt(data)

    prompt.append({"role": "user", "content": f"Review: {text}"})

    for i in range(num_retries):
        try:
            # Use the OpenAI API with GPT
            completion = openai.ChatCompletion.create(
                model=model, # Specify the GPT engine to be used for text completion
                temperature=0,
                max_tokens=10,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                messages=prompt
            )

            # Extract the predicted class from the API response
            answer = completion.choices[0].message.content

            if answer == "positive":
                answer = 1
            else:
                answer = 0
            return answer

        except Exception as e:
            if i == num_retries - 1:
                raise e
            else:
                print(f"Retrying in {initial_backoff * 2 ** i} seconds...")
                time.sleep(initial_backoff * 2 ** i)


if __name__ == "__main__":

    tqdm.pandas(desc="GPT-35-turbo label generation")

    test['gpt-label'] = test['text'].progress_apply(lambda x: get_sentiment(x, data, model=model, num_retries=num_retries, initial_backoff=initial_backoff))
    test.to_csv("data/processed/imdb/test.csv", index=False)




