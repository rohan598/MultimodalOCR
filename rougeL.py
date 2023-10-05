# code adapted from
# https://github.com/allenai/natural-instructions/blob/master/eval/automatic/evaluation.py
import string
import json
import os
from rouge_score import rouge_scorer
from rouge import Rouge
from transformers import AutoTokenizer


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

# def rouge(prediction, ground_truth, xlingual=False):
#     scorer = default_rouge_scorer
#     scores = scorer.score(prediction=prediction, target=ground_truth)
#     print(scores["rougeL"])
#     return scores["rougeL"].fmeasure


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_all_per_category(em_list, rougeL_list, category_dict):

    em = 100.0 * sum(em_list) / category_dict["total_cnt"]
    rougeL = 100.0 * sum(rougeL_list) / category_dict["total_cnt"]
    metrics = {"exact_match": em, "rougeL": rougeL}

    for k, v in category_dict.items():
        if k == "total_cnt" or len(v)<1:
            
            continue
        metrics[f"exact_match_{k}"] = 100.0 * sum([em_list[i] for i in v]) / len(v)
        metrics[f"rougeL_{k}"] = 100.0 * sum([rougeL_list[i] for i in v]) / len(v)

    metrics = {k: round(v, 4) for k, v in metrics.items()}

    return metrics

def compute_metrics(predictions, references, category_dict):
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    em, rougeL = [], []
    
    for i in range((len(predictions))):
        prediction = predictions[i]
        reference = references[i]
        if reference is not list:
            reference = [reference]
        em.append(metric_max_over_ground_truths(
            exact_match, prediction=prediction, ground_truths=reference
        ))
        rougeL.append(metric_max_over_ground_truths(
            rouge, prediction=prediction, ground_truths=reference
        ))

    metrics = compute_all_per_category(em, rougeL, category_dict)
    return metrics, em, rougeL


def run_rougeL(answer_path, args):
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
    results, em, rougeL = compute_metrics(predictions, references, category_dict)

    # updtae prediction list
    cnt = 0
    for pred in prediction_list:
        pred["exact_match"] = 1 if em[cnt]== True else 0
        pred["rougeL"] = rougeL[cnt]
        cnt+=1

    print("======== Overall Metrics ========")
    for metric, value in results.items():
        print(f"{metric}: {value}")
        all_results[f"{metric}"] = value

    return all_results, prediction_list
