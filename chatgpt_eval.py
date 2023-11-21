import os
import csv
import time 
import openai
import argparse
import pandas as pd
from tqdm import tqdm
import json
import cv2
from paddleocr import PaddleOCR
from common import convert_sample_to_description
from constants import RANKINGS_PROMPT

openai.api_key = os.getenv("OPENAI_API_KEY")
parser = argparse.ArgumentParser()

parser.add_argument('--gpt_version', choices=['gpt-3.5-turbo', 'gpt-4', 'gpt-3.5-turbo-16k'], default='gpt-3.5-turbo')
parser.add_argument('--input_filepath', type = str, default = 'chatgpt_feedback/without_dolly/test_pairwise_data.csv')
parser.add_argument('--save_feedback_filepath', type = str, default = None)
parser.add_argument('--start_index', type = int, default = 0)

args = parser.parse_args()

PROMPT_DICT = {
    "prompt_input": (
        "{instruction}\n\nOCR Text of the image:\n{input}"
    ),
    "prompt_no_input": (
        "{instruction}"
    ),
}

def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')
    print('Wrote {} records to {}'.format(len(data), output_path))


def get_reward(instruction, input, gt_answer, output_1, output_2):
    if str(input) == "":
        instruction = PROMPT_DICT['prompt_no_input'].format(instruction = instruction)
        prompt = RANKINGS_PROMPT.format(instruction = instruction, gt_answer = gt_answer, output_1 = output_1, output_2 = output_2)
    else:
        instruction = PROMPT_DICT['prompt_input'].format(instruction = instruction, input = input)
        prompt = RANKINGS_PROMPT.format(instruction = instruction, gt_answer = gt_answer,output_1 = output_1, output_2 = output_2)

    messages = [{"role": "user", "content": prompt}]

    return messages


def main():

    df = pd.read_csv(args.input_filepath)
    df = df.iloc[args.start_index:]

    chatgpt_eval_data = []
    cnt = 0
    reader = PaddleOCR(use_angle_cls=False, lang='en')
    for j in tqdm(range(len(df))):
        # if cnt == 20:
        #     break

        try:
            instruction = df.iloc[j]['question']
            # input = ""
            image_path = "/local1/rwadhawan/document_understanding/datasets/training/mplug_owl/test/images/" + df.iloc[j]['image_path'].split("/")[-1]
            image = cv2.imread(image_path)
            image = image[..., ::-1] 
            input = convert_sample_to_description(image, reader) 
            gt_answer = df.iloc[j]['gt_answer']
            output1 = df.iloc[j]['answer_model_i']
            output2 = df.iloc[j]['answer_model_j']
            completion = openai.ChatCompletion.create(
                model = args.gpt_version, 
                messages = get_reward(instruction, input, gt_answer, output1, output2))
            feedback_1 = completion['choices'][0]['message']['content']
            completion = openai.ChatCompletion.create(
                model = args.gpt_version, 
                messages = get_reward(instruction, input, gt_answer, output2, output1))
            feedback_2 = completion['choices'][0]['message']['content']
            if '(a)' in feedback_1 and '(b)' in feedback_2:
                feedback = '(a)'
            elif '(b)' in feedback_1 and '(a)' in feedback_2:
                feedback = '(b)'
            elif '(a)' in feedback_1 and '(a)' in feedback_2:
                feedback = '(c)'
            elif '(b)' in feedback_1 and '(b)' in feedback_2:
                feedback = '(c)'
            elif '(c)' in feedback_1 or '(c)' in feedback_2:
                feedback = '(c)'
            else:
                continue
            print(instruction)
            print(gt_answer)
            print(feedback_1, feedback_2, feedback)
            cnt+=1
            chatgpt_eval_data.append({
                "instruction":instruction,
                "gt_answer":gt_answer,
                "output_1":output1,
                "output_2":output2,
                "feedback_1":feedback_1,
                "feedback_2":feedback_2,
                "feedback":feedback
            })
        except:
            print('Sleeping...')
            time.sleep(5)
            chatgpt_eval_data.append({
                "instruction":instruction,
                "gt_answer":gt_answer,
                "output_1":output1,
                "output_2":output2,
                "feedback_1":feedback_1,
                "feedback_2":feedback_2,
                "feedback":"NA"
            })
        if cnt%15==0:
            print("interval sleeping")
            time.sleep(10)
    dump_jsonl(chatgpt_eval_data, args.save_feedback_filepath)

if __name__ == '__main__':
    main()