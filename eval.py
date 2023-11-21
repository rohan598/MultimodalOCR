import sys
sys.path.append('./models/MiniGPT4')
sys.path.append('./models/mPLUG_Owl')
import argparse
#from models.BLIP2.BLIP2 import BLIP2
import more_itertools
from tqdm import tqdm
import datetime
import os
import json
import re
import cv2
from paddleocr import PaddleOCR
from datasets.docit_dataset import DocitDataset
from datasets.vqa_dataset import textVQADataset, docVQADataset, ocrVQADataset, STVQADataset, ESTVQADataset, ChartQADataset, InfoVQADataset
from datasets.ocr_dataset import ocrDataset, IAMDataset, ReCTSDataset
from datasets.kie_dataset import SROIEDataset,FUNSDDataset,POIEDataset
# from datasets.formula_dataset import HMEDataset
# from models.lavis.lavis import lavis
from models.LLaVA.model_vqa import LLaVA
from models.LLaVAR.LLaVAR import LLaVAR
from models.mPLUG_Owl.pipeline.mPLUG import mPLUG
# from bard_api import Bard_Model
# from models.MiniGPT4.MiniGPT4 import MiniGPT4
# from models.OpenFlamingo.OpenFlamingo import OpenFlamingo
# from models.BLIP2.BLIP2 import BLIP2
# from models.InstructBLIP.InstructBLIP import InstructBLIP
import torch
import numpy as np
from PIL import Image
from common import convert_sample_to_description
from openai_api import openai_chat_completion
from similarity_score_docit import similarity_score

LATIN_PROMPT_TEMPLATE = """
You are asked to answer questions asked on a document image.

The answers to questions are short text spans taken verbatim from the document. This means that the answers comprise a set
of contiguous text tokens present in the document.
Document:
{document}

Question: {question}

Directly extract the answer of the question from the document with as few words as possible.

Answer:
"""

LATIN_PROMPT_IC_TEMPLATE = """
You are asked to answer questions asked on an image with text embedded in it. The answers to questions are short text spans taken verbatim from the image or require reasoning with the extracted. You will be given OCR text for the image, corresponding Image Caption and a Question. 

## Annotation Example
To help you understand the task, we provide some examples below.

Example 1:

OCR text for example 1:

Image caption for example 1:

Question for example 1: 

Answer for example 1:

Example 2:

OCR text for example 2:

Image caption for example 2:

Question for example 2: 

Answer for example 2:

Now is your turn. I will give you an example.
You should read the OCR text and Image Caption and then answer

OCR text:
{ocr_text}

Image caption:
{image_caption}

Question: 
{question}

Answer:
"""

def load_jsonl(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data 

def read_json_file(filepath):
    with open(filepath) as infile:
        data_dict = json.load(infile)
    return data_dict

def get_model(args):
    model = None

    if args.model_name=='llava':
        model = LLaVA(args.LLaVA_model_path, args.LLaVA_model_base)

    elif args.model_name=='llavar':
        model = LLaVAR(args.LLaVAR_model_path, args.device)
    
    elif "mPLUG" in args.model_name:
        model = mPLUG(args.mPLUG_model_name, args.device, args.mPLUG_lora_ckpt)
    
    elif args.model_name =="bard":
        pass
        # token = "cggvfgH1mJHd6PGW1Fm1CIWrjO6KTsqatcgtSoKha-s7af-VfAg4v1S7yCEM-FFxe_bZ4w."
        # model = Bard_Model(token=token)
    return model
def has_word(sentence, word):
    pattern = r"\b" + re.escape(word) + r"\b"
    match = re.search(pattern, sentence)
    if match:
        return True
    else:
        return False
def remove_special_chars(s):
    pattern = r"[^a-zA-Z0-9\s]"
    s = re.sub(pattern, "", s)
    return s

class VQAEval:
    def __init__(self):
        self.contractions = {
            "aint": "ain't",
            "arent": "aren't",
            "cant": "can't",
            "couldve": "could've",
            "couldnt": "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            "didnt": "didn't",
            "doesnt": "doesn't",
            "dont": "don't",
            "hadnt": "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            "hasnt": "hasn't",
            "havent": "haven't",
            "hed": "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            "hes": "he's",
            "howd": "how'd",
            "howll": "how'll",
            "hows": "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            "Im": "I'm",
            "Ive": "I've",
            "isnt": "isn't",
            "itd": "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            "itll": "it'll",
            "let's": "let's",
            "maam": "ma'am",
            "mightnt": "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            "mightve": "might've",
            "mustnt": "mustn't",
            "mustve": "must've",
            "neednt": "needn't",
            "notve": "not've",
            "oclock": "o'clock",
            "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            "shant": "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            "shouldve": "should've",
            "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd",
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            "somebodyll": "somebody'll",
            "somebodys": "somebody's",
            "someoned": "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            "someonell": "someone'll",
            "someones": "someone's",
            "somethingd": "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            "somethingll": "something'll",
            "thats": "that's",
            "thered": "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            "therere": "there're",
            "theres": "there's",
            "theyd": "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            "theyll": "they'll",
            "theyre": "they're",
            "theyve": "they've",
            "twas": "'twas",
            "wasnt": "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            "weve": "we've",
            "werent": "weren't",
            "whatll": "what'll",
            "whatre": "what're",
            "whats": "what's",
            "whatve": "what've",
            "whens": "when's",
            "whered": "where'd",
            "wheres": "where's",
            "whereve": "where've",
            "whod": "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            "wholl": "who'll",
            "whos": "who's",
            "whove": "who've",
            "whyll": "why'll",
            "whyre": "why're",
            "whys": "why's",
            "wont": "won't",
            "wouldve": "would've",
            "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            "yall": "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            "youd": "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            "youll": "you'll",
            "youre": "you're",
            "youve": "you've",
        }
        self.manualMap = {
            "none": "0",
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        self.articles = ["a", "an", "the"]

        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

    def evaluate(self, answer, gt_answers):
        
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()
        answer = self.processPunctuation(answer)
        answer = self.processDigitArticle(answer)

        if type(gt_answers)==list:
            for i in range(len(gt_answers)):
                gt_answers[i] = gt_answers[i].replace("\n", " ")
                gt_answers[i] = gt_answers[i].replace("\t", " ")
                gt_answers[i] = gt_answers[i].strip()
                gt_answers[i] = self.processPunctuation(gt_answers[i])
                gt_answers[i] = self.processDigitArticle(gt_answers[i])
                if has_word(answer, gt_answers[i]):
                    return 1
            return 0
        else:
            gt_answers = gt_answers.replace("\n", " ")
            gt_answers= gt_answers.replace("\t", " ")
            gt_answers = gt_answers.strip()
            gt_answers = self.processPunctuation(gt_answers)
            gt_answers = self.processDigitArticle(gt_answers)

            if has_word(answer, gt_answers):
                return 1
            else:
                return 0

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(self.commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = " ".join(outText)
        return outText

def get_prompt_input(batch, reader):
    image = cv2.imread(batch["image_path"])
    image = image[..., ::-1] 
    # image  = cv2.resize(image, (224, 224),
    #            interpolation = cv2.INTER_AREA)
    doc_text = convert_sample_to_description(image, reader)
    prompt_input = LATIN_PROMPT_TEMPLATE.format(document=doc_text, question=batch["question"])
    return prompt_input

def get_prompt_input_v2(batch, reader, image_caption):
    image = cv2.imread(batch["image_path"])
    image = image[..., ::-1] 
    # image  = cv2.resize(image, (224, 224),
    #            interpolation = cv2.INTER_AREA)
    ocr_text = convert_sample_to_description(image, reader)
    prompt_input = LATIN_PROMPT_IC_TEMPLATE.format(ocr_text=ocr_text, image_caption = image_caption, question=batch["question"])
    return prompt_input

def evaluate_VQA(
    model,
    dataset,
    model_name,
    dataset_name,
    time,
    image_caption_filepath="",
    batch_size=1,
    answer_path='./answers',
    no_pred = False,
    conv_mode = "llava_v1",
    temperature = 0.2,
    top_p = None, 
    num_beams = 1
):

    if no_pred==False:
        predictions=[]
        # reader = None
        cnt = 0
        reader = PaddleOCR(use_angle_cls=True, lang='en')
        for batch in more_itertools.chunked(
            tqdm(dataset, desc="Running inference"), batch_size
        ):
            batch = batch[0]
            if "gpt" in model_name:
                if "ic" in model_name:
                    image_caption_data = read_json_file(image_caption_filepath)
                    prompt_input = get_prompt_input_v2(batch, reader, image_caption_data[cnt]["answer"])
                else:
                    prompt_input = get_prompt_input(batch, reader)
                print(f"prompt_input: {prompt_input}")
                output = openai_chat_completion(prompt_input, model_name=model_name, max_tokens_to_sample = 512, stop="\n\n", temperature=0)
            elif "llava" in model_name:
                output = model.generate(image=batch['image_path'], 
                                    question=batch["question"],
                                    conv_mode=conv_mode, 
                                    temperature = temperature, 
                                    top_p = top_p, 
                                    num_beams = num_beams)
            else:
                output = model.generate(image=batch['image_path'], question=batch['question'])
            cnt+=1
            answer_dict={'question':batch['question'], 'answer':output, 
            'gt_answers':batch['gt_answers'], 'image_path':batch['image_path'],
            'model_name':model_name}
            print("MMOCR output", output)
            predictions.append(answer_dict)
        
        answer_dir = os.path.join(answer_path, f"{model_name}_{time}_{args.train_config}")
        os.makedirs(answer_dir, exist_ok=True)

        answer_path = os.path.join(answer_dir, f"{dataset_name}.json")
        with open(answer_path, "w") as f:
            f.write(json.dumps(predictions, indent=4))

    eval = VQAEval()
    correct = []
    with open(answer_path, 'r') as f:
        dict = json.load(f)
        for i in range(len(dict)):
            gt_answers = dict[i]['gt_answers']
            answer = dict[i]['answer']
            if eval.evaluate(answer,gt_answers)==1:
                correct.append(1)
            else:
                correct.append(0)

    cnt = 0
    for pred in dict:
        pred["exact_match"] = correct[cnt]
        cnt+=1 

    print(f'{dataset_name}:{float(sum(correct))/len(correct)}')
    return float(sum(correct))/len(correct), dict, answer_path

def evaluate_OCR(
    model,
    dataset,
    model_name,
    dataset_name,
    time,
    question='what is written in the image?',
    batch_size=1,
    answer_path='./answers',
    no_pred = False,
    conv_mode = "llava_v1",
    temperature = 0.2,
    top_p = None, 
    num_beams = 1
):
    if no_pred == False:
        predictions=[]
        # reader = None
        reader = PaddleOCR(use_angle_cls=True, lang='en')
        for batch in more_itertools.chunked(
            tqdm(dataset, desc="Running inference"), batch_size
        ):
            batch = batch[0]
            if "latin" in model_name:
                image = cv2.imread(batch["image_path"])
                image = image[..., ::-1] 
                output = convert_sample_to_description(image, reader)
            elif "llava" in model_name:
                output = model.generate(image=batch['image_path'], 
                                    question=batch["question"],
                                    conv_mode=conv_mode, 
                                    temperature = temperature, 
                                    top_p = top_p, 
                                    num_beams = num_beams)
            else:
                output = model.generate(image=batch['image_path'], question=question)

            answer_dict={'question':question, 'answer':output, 
            'gt_answers':batch['gt_answers'], 'image_path':batch['image_path'],
            'model_name':model_name}
            predictions.append(answer_dict)
       
        answer_dir = os.path.join(answer_path, f"{model_name}_{time}_{args.train_config}")
        os.makedirs(answer_dir, exist_ok=True)
        answer_path = os.path.join(answer_dir, f"{dataset_name}.json")
        with open(answer_path, "w") as f:
            f.write(json.dumps(predictions, indent=4))
    correct = []
    with open(answer_path, 'r') as f:
        dict = json.load(f)
        for i in range(len(dict)):
            gt_answers = dict[i]['gt_answers']
            answer = dict[i]['answer']
            gt_answers = remove_special_chars(gt_answers).lower()
            answer = remove_special_chars(answer).lower()
            if has_word(answer, gt_answers):
                correct.append(1)
            else:
                correct.append(0)

    cnt = 0
    for pred in dict:
        pred["exact_match"] = correct[cnt]
        cnt+=1 

    print(f'{dataset_name}:{float(sum(correct))/len(correct)}')
    return float(sum(correct))/len(correct), dict, answer_path

def prepare_similarity_score(
    model,
    dataset,
    model_name,
    dataset_name,
    time,
    image_caption_filepath,
    batch_size=1,
    answer_path='./answers',
    conv_mode = "llava_v1",
    temperature = 0.2,
    top_p = None, 
    num_beams = 1
):
    predictions=[]
    reader = PaddleOCR(use_angle_cls=True, lang='en')
    cnt = 0

    if model is None:
        if model_name!="gpt-35":
            model_name = "gpt-35"
            inf_type = "ocr_ic"
            image_caption_data = read_json_file(image_caption_filepath)
        else:
            inf_type = "reg"

    for batch in more_itertools.chunked(
        tqdm(dataset, desc="Running inference"), batch_size
    ):
        batch = batch[0]
        if "gpt" in model_name:
            prompt_input = get_prompt_input_v2(batch, reader, image_caption_data[cnt]["answer"])
            print(f"prompt_input: {prompt_input}")
            output = openai_chat_completion(prompt_input, model_name=model_name, max_tokens_to_sample = 512, stop="\n\n", temperature=0)

        elif "llava" in model_name:
            output = model.generate(image=batch['image_path'], 
                                    question=batch["question"],
                                    conv_mode=conv_mode, 
                                    temperature = temperature, 
                                    top_p = top_p, 
                                    num_beams = num_beams)
        else:
            output = model.generate(image=batch['image_path'], question=batch["question"])
        
        cnt+=1

        answer_dict={'question':batch["question"], 'answer':output, 
        'gt_answers':batch["gt_answers"], 'image_path':batch['image_path'],
        'model_name':model_name}
        predictions.append(answer_dict)

    if model is None:
        answer_dir = os.path.join(answer_path, f"{model_name}_{inf_type}_{time}")
    else: 
        answer_dir = os.path.join(answer_path, f"{model_name}_{time}_{args.train_config}")

    os.makedirs(answer_dir, exist_ok=True)
    answer_path = os.path.join(answer_dir, f"{dataset_name}.json")
    with open(answer_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))

    return answer_path

def bard_pred(
    model,
    dataset,
    model_name,
    dataset_name,
    time,
    batch_size=1,
    answer_path='./answers'
):
    predictions=[]
    if os.path.exists(answer_path):
        with open(answer_path) as fin:
            bard_pred_data = json.load(fin)

        for data_item in bard_pred_data:
           if data_item["answer"] == "":
            # output = model.get_response(data_item['image_path'], data_item["question"])
            # print("output: ", output)
            # data_item["answer"]  = output
            print(data_item["question"])
            print(data_item["image_path"])
            print()
                
        # with open(answer_path, "w") as f:
        #     f.write(json.dumps(bard_pred_data, indent=4))
    else:
        for batch in more_itertools.chunked(
            tqdm(dataset, desc="Running inference"), batch_size
        ):
            batch = batch[0]
            output = model.get_response(batch['image_path'], batch["question"])
            print("output: ", output)

            answer_dict={'question':batch["question"], 'answer':output, 
            'gt_answers':batch["gt_answers"], 'image_path':batch['image_path'],
            'model_name':model_name}
            predictions.append(answer_dict)

        answer_dir = os.path.join(answer_path, f"{model_name}_{time}")

        os.makedirs(answer_dir, exist_ok=True)
        answer_path = os.path.join(answer_dir, f"{dataset_name}.json")
        with open(answer_path, "w") as f:
            f.write(json.dumps(predictions, indent=4))         

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    #OCR datasets
    parser.add_argument("--ocr_dir_path", type=str, default="./data")
    parser.add_argument("--ocr_dataset_name", type=str, default="IIIT5K svt IC13_857 IC15_1811 svtp ct80 cocotext ctw totaltext HOST WOST WordArt CAR_A")
    #IAM
    parser.add_argument("--IAM_dir_path", type=str, default="./data/IAM")

    #textVQA
    parser.add_argument("--textVQA_image_dir_path", type=str, default="./data/textVQA/train_images")
    parser.add_argument("--textVQA_ann_path", type=str, default="./data/textVQA/TextVQA_0.5.1_val.json")

    #docVQA
    parser.add_argument("--docVQA_image_dir_path", type=str, default="./data/docVQA/val")
    parser.add_argument("--docVQA_ann_path", type=str, default="./data/docVQA/val/val_v1.0.json")

    #ocrVQA
    parser.add_argument("--ocrVQA_image_dir_path", type=str, default="./data/ocrVQA/images")
    parser.add_argument("--ocrVQA_ann_path", type=str, default="./data/ocrVQA/dataset.json")

    #STVQA
    parser.add_argument("--STVQA_image_dir_path", type=str, default="./data/STVQA")
    parser.add_argument("--STVQA_ann_path", type=str, default="./data/STVQA/train_task_3.json")
    #ESTVQA
    parser.add_argument("--ESTVQA_image_dir_path", type=str, default="./data/ESTVQA/images/train")
    parser.add_argument("--ESTVQA_CN_ann_path", type=str, default="./data/ESTVQA/annotations/cn_train.json")
    parser.add_argument("--ESTVQA_EN_ann_path", type=str, default="./data/ESTVQA/annotations/en_train.json")

    #Infograhics VQA
    parser.add_argument("--infoVQA_image_dir_path", type=str, default="./infographicsvqa/infographicsvqa_images")
    parser.add_argument("--infoVQA_ann_path", type=str, default="./infographicsvqa/infographicsvqa_qas/infographicsVQA_val_v1.0_withQT")

    #Screen2Words
    parser.add_argument("--S2W_image_dir_path", type=str, default="./test/png")
    parser.add_argument("--S2W_ann_path", type=str, default="./test/test_human.json")

    #FUNSD
    parser.add_argument("--FUNSD_dir_path", type=str, default="./data/FUNSD/testing_data/annotations")

    # DOCIT
    parser.add_argument("--docit_test_filepath", default="")

    # image caption filepath
    parser.add_argument("--image_caption_filepath", default="")
    
    #result_path
    parser.add_argument("--answer_path", type=str, default="./answers")

    parser.add_argument(
        "--eval_textVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on textVQA."
    )
    parser.add_argument(
        "--eval_docVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on docVQA."
    )
    parser.add_argument(
        "--eval_ocrVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on ocrVQA."
    )
    parser.add_argument(
        "--eval_STVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on STVQA."
    )
    parser.add_argument(
        "--eval_ESTVQA_EN",
        action="store_true",
        default=False,
        help="Whether to evaluate on ESTVQA_EN."
    )
    parser.add_argument(
        "--eval_infoVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on Infographics VQA."
    )
    parser.add_argument(
        "--eval_S2W",
        action="store_true",
        default=False,
        help="Whether to evaluate on Screen2Words."
    )
    parser.add_argument(
        "--eval_FUNSD",
        action="store_true",
        default=False,
        help="Whether to evaluate on FUNSD."
    )
    parser.add_argument(
        "--eval_IAM",
        action="store_true",
        default=False,
        help="Whether to evaluate on IAM (handwritten)."
    )

    parser.add_argument(
        "--eval_ocr",
        action="store_true",
        default=False,
        help="Whether to evaluate on ocr."
    )

    parser.add_argument("--eval_sim_score",
                        action="store_true",
                        default=False,
                        help="Whether to evaluate on rouge L and bleurt.")
    
    parser.add_argument(
        "--eval_bard",
        action="store_true",
        default=False,
        help="if prediction already done"
    )  

    parser.add_argument(
        "--eval_ocr_ic",
        action="store_true",
        default=False,
        help="if prediction already done"
    )  
    
    parser.add_argument(
        "--eval_all",
        action="store_true",
        default=False,
        help="Whether to evaluate all datasets"
    )
    parser.add_argument(
        "--no_pred",
        action="store_true",
        default=False,
        help="if prediction already done"
    )  

    parser.add_argument(
        "--no_pred_filename",
        default="tmp",
        help="if prediction already done new filename"
    )  

    #LLaVA
    parser.add_argument("--LLaVA_model_path", type=str, default="")
    parser.add_argument("--LLaVA_model_base", type=str, default="")
    parser.add_argument("--LLaVA_model_conv_mode", type=str, default="llava_v1")
    parser.add_argument("--LLaVA_model_temperature",  type=float, default=0.2)
    parser.add_argument("--LLaVA_model_top_p",  type=int, default=None)
    parser.add_argument("--LLaVA_model_num_beams",  type=int, default=1)

    #LLaVAR
    parser.add_argument("--LLaVAR_model_path", type=str, default="")

    #mPLUG
    parser.add_argument("--mPLUG_model_name", type=str, default="MAGAer13/mplug-owl-llama-7b")
    parser.add_argument("--mPLUG_lora_ckpt", type=str, default="")
    parser.add_argument("--mPLUG_tokenizer_path", type=str, default="./models/mPLUG_Owl/model_weight/tokenizer.model")

    parser.add_argument("--model_name", type=str, default="mPLUG")#mPLUG,miniGPT4,LLaVA
    parser.add_argument("--train_config", type=str, default="abc_1e1")#mPLUG,
    parser.add_argument("--device", type=str, default="cuda:0")#2,3,7
    args = parser.parse_args()
    return args

def main(args):
    np.random.seed(0)
    max_sample_num = 5000
    if args.no_pred==False:
        model = get_model(args)
    else:
        model = None
    '''ocr_dataset_name=['IIIT5K','svt','IC13_857','IC15_1811','svtp','ct80',
                  'cocotext','ctw','totaltext','HOST','WOST','WordArt']'''
    ocr_dataset_name = args.ocr_dataset_name.split()
    result = {}
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    start_time = datetime.datetime.now()

    if args.eval_textVQA or args.eval_all:
        dataset = textVQADataset(args.textVQA_image_dir_path, args.textVQA_ann_path)

        answer_path = args.answer_path
        if args.no_pred:
            answer_path = os.path.join(answer_path, "textVQA.json")

        acc, prediction_list, answer_path = evaluate_VQA(model, dataset, args.model_name, 'textVQA', time, image_caption_filepath= args.image_caption_filepath, answer_path = answer_path, no_pred = args.no_pred, conv_mode=args.LLaVA_model_conv_mode, temperature=args.LLaVA_model_temperature, top_p = args.LLaVA_model_top_p, num_beams = args.LLaVA_model_num_beams)
    
        result['textVQA'] = acc
        with open(answer_path, "w") as f:
                        f.write(json.dumps(prediction_list, indent=4))

        print("textVQA eval complete")

    if args.eval_docVQA or args.eval_all:
        dataset = docVQADataset(args.docVQA_image_dir_path, args.docVQA_ann_path)
        answer_path = args.answer_path
        if args.no_pred:
            answer_path = os.path.join(answer_path, "docVQA.json")
        
        acc, prediction_list, answer_path = evaluate_VQA(model, dataset, args.model_name, 'docVQA', time, image_caption_filepath= args.image_caption_filepath, answer_path = answer_path, no_pred = args.no_pred, conv_mode=args.LLaVA_model_conv_mode, temperature=args.LLaVA_model_temperature, top_p = args.LLaVA_model_top_p, num_beams = args.LLaVA_model_num_beams)
        
        result['docVQA'] = acc

        with open(answer_path, "w") as f:
                f.write(json.dumps(prediction_list, indent=4))

        print("docVQA eval complete")
        
    #Due to network issues, it's difficult to download the entire OCR-VQA dataset. 
    # Therefore, we will extract the first 5000 questions for testing.

    ## my comment - starts
    if args.eval_ocrVQA or args.eval_all:
        dataset = ocrVQADataset(args.ocrVQA_image_dir_path, args.ocrVQA_ann_path)
        
        answer_path = args.answer_path
        if args.no_pred:
            answer_path = os.path.join(answer_path, "ocrVQA.json")

        dataset = torch.utils.data.Subset(dataset, range(max_sample_num))
        acc, prediction_list, answer_path = evaluate_VQA(model, dataset, args.model_name, 'ocrVQA', time, image_caption_filepath= args.image_caption_filepath, answer_path = answer_path, no_pred = args.no_pred, conv_mode=args.LLaVA_model_conv_mode, temperature=args.LLaVA_model_temperature, top_p = args.LLaVA_model_top_p, num_beams = args.LLaVA_model_num_beams)
        result['ocrVQA'] = acc
    
        with open(answer_path, "w") as f:
                f.write(json.dumps(prediction_list, indent=4))

        print("ocrVQA eval complete")
    
    if args.eval_STVQA or args.eval_all:
        dataset = STVQADataset(args.STVQA_image_dir_path, args.STVQA_ann_path)
        dataset = torch.utils.data.Subset(dataset, range(max_sample_num))
        answer_path = args.answer_path
        if args.no_pred:
            answer_path = os.path.join(answer_path, "STVQA.json")

        acc, prediction_list, answer_path = evaluate_VQA(model, dataset, args.model_name, 'STVQA', time, image_caption_filepath= args.image_caption_filepath, answer_path = answer_path, no_pred = args.no_pred, conv_mode=args.LLaVA_model_conv_mode, temperature=args.LLaVA_model_temperature, top_p = args.LLaVA_model_top_p, num_beams = args.LLaVA_model_num_beams)

        result['STVQA'] = acc

        with open(answer_path, "w") as f:
                f.write(json.dumps(prediction_list, indent=4))

        print("STVQA eval complete")

    if args.eval_ESTVQA_EN or args.eval_all:
        dataset = ESTVQADataset(args.ESTVQA_image_dir_path, args.ESTVQA_EN_ann_path)
        dataset = torch.utils.data.Subset(dataset, range(max_sample_num))
        answer_path = args.answer_path
        
        if args.no_pred:
            answer_path = os.path.join(answer_path, "ESTVQA_EN.json")
        
        acc, prediction_list, answer_path = evaluate_VQA(model, dataset, args.model_name, 'ESTVQA_EN', time, image_caption_filepath= args.image_caption_filepath, answer_path = answer_path, no_pred = args.no_pred, conv_mode=args.LLaVA_model_conv_mode, temperature=args.LLaVA_model_temperature, top_p = args.LLaVA_model_top_p, num_beams = args.LLaVA_model_num_beams)

        result['ESTVQA_EN'] = acc

        with open(answer_path, "w") as f:
                f.write(json.dumps(prediction_list, indent=4))

        print("ESTVQA_EN eval complete")


    if args.eval_infoVQA or args.eval_all:
        dataset = InfoVQADataset(args.infoVQA_image_dir_path, args.infoVQA_ann_path)
        answer_path = args.answer_path
        if args.no_pred:
            answer_path = os.path.join(answer_path, "infoVQA.json")
        acc, prediction_list, answer_path = evaluate_VQA(model, dataset, args.model_name, 'infoVQA', time, image_caption_filepath= args.image_caption_filepath,  answer_path = answer_path, no_pred = args.no_pred, conv_mode=args.LLaVA_model_conv_mode, temperature=args.LLaVA_model_temperature, top_p = args.LLaVA_model_top_p, num_beams = args.LLaVA_model_num_beams)
        result['infoVQA'] = acc

        with open(answer_path, "w") as f:
                f.write(json.dumps(prediction_list, indent=4))

        print("InfographicsVQA eval complete")

    if args.eval_S2W:
        dataset = ESTVQADataset(args.ESTVQA_image_dir_path, args.ESTVQA_EN_ann_path)
        dataset = torch.utils.data.Subset(dataset, range(max_sample_num))
        answer_path = args.answer_path
        if args.no_pred:
            answer_path = os.path.join(answer_path, "S2W.json")
        acc, prediction_list, answer_path = evaluate_VQA(model, dataset, args.model_name, 'S2W', time, image_caption_filepath= args.image_caption_filepath, answer_path = answer_path, no_pred = args.no_pred, conv_mode=args.LLaVA_model_conv_mode, temperature=args.LLaVA_model_temperature, top_p = args.LLaVA_model_top_p, num_beams = args.LLaVA_model_num_beams)
        result['S2W'] = acc

        with open(answer_path, "w") as f:
                f.write(json.dumps(prediction_list, indent=4))

        print("Screen2Words eval complete")

    if args.eval_FUNSD or args.eval_all:
        dataset = FUNSDDataset(args.FUNSD_dir_path)
        answer_path = args.answer_path
        if args.no_pred:
            answer_path = os.path.join(answer_path, "FUNSD.json")
        acc, prediction_list, answer_path = evaluate_VQA(model, dataset, args.model_name, 'FUNSD', time, image_caption_filepath= args.image_caption_filepath, answer_path = answer_path, no_pred = args.no_pred, conv_mode=args.LLaVA_model_conv_mode, temperature=args.LLaVA_model_temperature, top_p = args.LLaVA_model_top_p, num_beams = args.LLaVA_model_num_beams)
        result['FUNSD'] = acc

        with open(answer_path, "w") as f:
                f.write(json.dumps(prediction_list, indent=4))

        print("FUNSD eval complete")
    
    if args.eval_IAM or args.eval_all:
        dataset = IAMDataset(args.IAM_dir_path)
        dataset = torch.utils.data.Subset(dataset, range(3000))
        answer_path = args.answer_path
        
        if args.no_pred:
            answer_path = os.path.join(answer_path, "IAM.json")
        
        acc, prediction_list, answer_path = evaluate_OCR(model, dataset, args.model_name, 'IAM', time, answer_path = answer_path, no_pred = args.no_pred, conv_mode=args.LLaVA_model_conv_mode, temperature=args.LLaVA_model_temperature, top_p = args.LLaVA_model_top_p, num_beams = args.LLaVA_model_num_beams)

        result['IAM'] = acc

        with open(answer_path, "w") as f:
                f.write(json.dumps(prediction_list, indent=4))

        print("FUNSD eval complete")

    if args.eval_ocr or args.eval_all:
        for i in range(len(ocr_dataset_name)):
            dataset = ocrDataset(args.ocr_dir_path, ocr_dataset_name[i])
            answer_path = args.answer_path
            if args.no_pred:
                answer_path = os.path.join(answer_path, f"{ocr_dataset_name[i]}.json")
            acc, prediction_list, answer_path = evaluate_OCR(model, dataset, args.model_name, ocr_dataset_name[i], time, answer_path = answer_path, no_pred = args.no_pred, conv_mode=args.LLaVA_model_conv_mode, temperature=args.LLaVA_model_temperature, top_p = args.LLaVA_model_top_p, num_beams = args.LLaVA_model_num_beams)
            result[ocr_dataset_name[i]] = acc

            with open(answer_path, "w") as f:
                f.write(json.dumps(prediction_list, indent=4))
            print(f"OCR dataset eval complete : {ocr_dataset_name[i]}")

    if args.eval_sim_score:
        answer_path = args.answer_path
        if args.no_pred == False:
            dataset = DocitDataset(args.docit_test_filepath)
            answer_path = prepare_similarity_score(model, dataset, args.model_name, "Docit", time, args.image_caption_filepath, answer_path = args.answer_path, conv_mode=args.LLaVA_model_conv_mode, temperature=args.LLaVA_model_temperature, top_p = args.LLaVA_model_top_p, num_beams = args.LLaVA_model_num_beams)
        else:
            answer_path = os.path.join(answer_path, "Docit.json")
        rl_results, prediction_list = similarity_score(answer_path, args)
        result['rougeL'] = rl_results["rougeL"]
        result['bleurt'] = rl_results["bleurt"]
        result['exact_match'] = rl_results["exact_match"]

        result['rougeL_documents'] = rl_results["rougeL_documents"]
        result['bleurt_documents'] = rl_results["bleurt_documents"]
        result['exact_match_documents'] = rl_results["exact_match_documents"]
        
        result['rougeL_screenshots'] = rl_results["rougeL_screenshots"]
        result['bleurt_screenshots'] = rl_results["bleurt_screenshots"]
        result['exact_match_screenshots'] = rl_results["exact_match_screenshots"]

        result['rougeL_text_recognition'] = rl_results["rougeL_text_recognition"]
        result['bleurt_text_recognition'] = rl_results["bleurt_text_recognition"]
        result['exact_match_text_recognition'] = rl_results["exact_match_text_recognition"]

        result['rougeL_llavar'] = rl_results["rougeL_llavar"]
        result['bleurt_llavar'] = rl_results["bleurt_llavar"]
        result['exact_match_llavar'] = rl_results["exact_match_llavar"]

        with open(answer_path, "w") as f:
            f.write(json.dumps(prediction_list, indent=4))
        print("Similarity score eval complete")   
    
    run_only_model = False

    if args.eval_bard:
        run_only_model = True
        answer_path = args.answer_path
        dataset = DocitDataset(args.docit_test_filepath, is_bard = True)
        bard_pred(model, dataset, args.model_name, "Docit", time, answer_path = args.answer_path)
        print("BARD prediction complete")   
        
    print(f"Eval Run time {(datetime.datetime.now() - start_time).total_seconds()}")
    ## my comment - ends
    if run_only_model == False:
        if args.no_pred == False:
            result_path = os.path.join(os.path.join(args.answer_path, f"{args.model_name}_{time}_{args.train_config}"), 'result.json')
        else:
            result_dir = args.answer_path
            os.makedirs(result_dir, exist_ok=True)
            result_path = os.path.join(result_dir, "result.json")

        with open(result_path, "w") as f:
            f.write(json.dumps(result, indent=4))

if __name__ == "__main__":
    args = parse_args()
    main(args)