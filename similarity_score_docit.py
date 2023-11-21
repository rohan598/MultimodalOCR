# code adapted from
# https://github.com/allenai/natural-instructions/blob/master/eval/automatic/evaluation.py
import string
import json
import os
from rouge_score import rouge_scorer
from rouge import Rouge
from transformers import AutoTokenizer
from bleurt import score as bleurt_score
import re

class TextProcess:
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
    
    def process_text(self, answer):
        
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()
        answer = self.processPunctuation(answer)
        answer = self.processDigitArticle(answer)

        return answer




class GPTTokenizer:
    gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2", max_length=1e5)

    def tokenize(self, s):
        tokens = self.gpt_tokenizer.tokenize(s)
        # GPT2 uses Byte-level BPE, which will include space as part of the word. 
        # But for the first word of a sentence, there is no space before it. 
        # So, we remove all the added spaces ("Ġ"). 
        tokens = [t.lstrip("Ġ") for t in tokens]
        return tokens

default_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
xlingual_tokenizer = GPTTokenizer()
xlingual_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], tokenizer=xlingual_tokenizer) 

# adapted the flowing from Squad v1.1 evaluation, without removing the articles.
def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def exact_match(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def rouge(prediction, ground_truth):
    prediction = prediction.strip()
    ground_truth = ground_truth.strip()
    
    if len(ground_truth)==0 or len(prediction)==0:
        return 0
    
    prediction = [prediction]
    ground_truth = [ground_truth]
    rouge = Rouge(metrics=["rouge-l"])
    scores = rouge.get_scores(prediction, ground_truth, avg=True)
    rougeL = scores['rouge-l']['f']

    return rougeL

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_all_per_category(em_list, rougeL_list, bleurt_scores_list, category_dict):

    em = 100.0 * sum(em_list) / category_dict["total_cnt"]
    rougeL = 100.0 * sum(rougeL_list) / category_dict["total_cnt"]
    bleurt_score_overall = 100.0 * sum(bleurt_scores_list) / category_dict["total_cnt"]
    metrics = {"exact_match": em, "rougeL": rougeL, "bleurt": bleurt_score_overall}

    for k, v in category_dict.items():
        if k == "total_cnt" or len(v)<1:
            
            continue
        metrics[f"exact_match_{k}"] = 100.0 * sum([em_list[i] for i in v]) / len(v)
        metrics[f"rougeL_{k}"] = 100.0 * sum([rougeL_list[i] for i in v]) / len(v)
        metrics[f"bleurt_{k}"] = 100.0 * sum([bleurt_scores_list[i] for i in v]) / len(v)

    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics

def compute_metrics(predictions, references, category_dict):
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    em, rougeL, bleurt_scores = [], [], []
    bleurt_scorer = bleurt_score.BleurtScorer("/local1/rwadhawan/document_understanding/models/BLEURT-20")
    tp = TextProcess()
    for i in range((len(predictions))):
        prediction = tp.process_text(predictions[i])
        reference = tp.process_text(references[i])
        if reference is not list:
            reference = [reference]
    
        em.append(metric_max_over_ground_truths(
            exact_match, prediction=prediction, ground_truths=reference
        ))
        rougeL.append(metric_max_over_ground_truths(
            rouge, prediction=prediction, ground_truths=reference
        ))
        prediction = [prediction]
        bleurt_scores.append(max(bleurt_scorer.score(references=reference, candidates=prediction
        )[0],0))

    metrics = compute_all_per_category(em, rougeL, bleurt_scores, category_dict)
    return metrics, em, rougeL, bleurt_scores


def similarity_score(answer_path, args):
    eval_instances = {}
    with open(args.docit_test_filepath, 'r', encoding='utf-8') as fin:
        cnt = 0
        for line in fin:
            instance = json.loads(line)
            # image_path = instance['image'][0] if type(instance['image']) is list else instance['image']
            eval_instances[cnt] = instance
            cnt+=1
    all_predictions = {}
    with open(answer_path) as fin:
        prediction_list = json.load(fin)
        cnt = 0
        for prediction in prediction_list:
            all_predictions[cnt] = prediction["answer"]
            cnt+=1

    all_results = {}

    instance_ids = [id for id, instance in eval_instances.items()]
    references = [eval_instances[id]["text"].split("Human:")[-1].split("AI:")[-1] for id in instance_ids]
    predictions = []
    missing_predictions = []
    category_dict = {"documents":[], "screenshots":[], "llavar":[], "text_recognition":[]}
    cnt = 0
    for id in instance_ids:
        if id in all_predictions:
            predictions.append(all_predictions[id])
        else:
            missing_predictions.append(id)
            predictions.append("")
            print(eval_instances[id]["category"])
        category_dict[eval_instances[id]["category"]].append(cnt)
        cnt+=1
    if missing_predictions:
        print(f"No prediction for {len(missing_predictions)} instances. Use empty string as prediction.")

    category_dict["total_cnt"] = len(all_predictions)
    results, em, rougeL, bleurt_scores = compute_metrics(predictions, references, category_dict)

    # updtae prediction list
    cnt = 0
    for pred in prediction_list:
        pred["exact_match"] = 1 if em[cnt]== True else 0
        pred["rougeL"] = rougeL[cnt]
        pred["bleurt"] = bleurt_scores[cnt]
        cnt+=1

    print("======== Overall Metrics ========")
    for metric, value in results.items():
        print(f"{metric}: {value}")
        all_results[f"{metric}"] = value

    return all_results, prediction_list
