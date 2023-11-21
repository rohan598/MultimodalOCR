import sys
sys.path.append('./models/mPLUG_Owl')
import os
import more_itertools
from tqdm import tqdm
import json
import datetime
import argparse

from datasets.docit_dataset import DocitDataset
from models.mPLUG_Owl.pipeline.mPLUG import mPLUG

def get_model(args):
    model = None
    if "mPLUG" in args.model_name:
        model = mPLUG(args.mPLUG_model_name, args.device, args.mPLUG_lora_ckpt)
    
    return model

def gen_captions(
    model,
    dataset,
    model_name,
    src_dataset="",
    batch_size=1,
    answer_path='./answers',
    filename = "image_captions.json",
    template=None
):
    predictions=[]
    cnt= 0
    # question = "Generate a caption for the given image of a document. You can use text within the image for better understanding."
    for batch in more_itertools.chunked(
        tqdm(dataset, desc="Running inference"), batch_size
    ):
        batch = batch[0]
        if template == "generic":
            question = "Describe the image? You can use text within the image for better understanding."
        elif template == "specific":
            if batch["category"] == "documents":
                question = "Generate a caption for the given image of a document. You can use text within the image for better understanding."
            elif batch["category"] == "screenshots":
                question = "Generate a caption for the given image of a mobile application UI screenshot or website screenshot. You can use text within the image for better understanding."
            elif batch["category"] == "text_recognition":
                question = "Generate a caption for the given image of a word embedded in it. You can use text within the image for better understanding."
            else:
                question = "Generate a caption for the given image of a book cover, poster, meme, infographic or a related visual artifact. You can use text within the image for better understanding."
        image_path = os.path.join("/local1/rwadhawan/document_understanding/datasets/training/mplug_owl/test/images", batch['image_path'].split("/")[-1])
        output = model.generate(image=image_path, question=question)
        answer_dict={'question':question, 'answer':output, 'image_path':image_path, 'model_name':model_name}
        predictions.append(answer_dict)

    answer_dir = os.path.join(answer_path, f"{model_name}_{src_dataset}")

    os.makedirs(answer_dir, exist_ok=True)
    answer_path = os.path.join(answer_dir, filename)
    print(answer_path)
    with open(answer_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))

    return answer_path

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--model_name", type=str, default="mPLUG")
    # DOCIT
    parser.add_argument("--docit_test_filepath", default="")
    
    #result_path
    parser.add_argument("--answer_path", type=str, default="./answers")

    #mPLUG
    parser.add_argument("--mPLUG_model_name", type=str, default="MAGAer13/mplug-owl-llama-7b")
    parser.add_argument("--mPLUG_lora_ckpt", type=str, default="")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    model = get_model(args)
    dataset = DocitDataset(args.docit_test_filepath)
    
    src_dataset = args.answer_path.split("/")[-1].split(".")[0]
    # create two version of image captions
    # version 1 generic template
    # answer_path = gen_captions(model, dataset, args.model_name, src_dataset=src_dataset, answer_path = args.answer_path, filename="image_captions_generic.json", template="generic")

    # version 2 specific template
    answer_path = gen_captions(model, dataset, args.model_name, src_dataset=src_dataset, answer_path = args.answer_path, filename="image_captions_specific.json", template="specific")