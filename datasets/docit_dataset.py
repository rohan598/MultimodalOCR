from torch.utils.data import Dataset
import os
import json

def load_jsonl(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data 

class DocitDataset(Dataset):
    def __init__(
        self,
        docit_jsonl_filepath
    ):
        
        self.data = load_jsonl(docit_jsonl_filepath)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['text'].split("Human:")[-1].split("AI:")[0]
        answers = self.data[idx]['text'].split("Human:")[-1].split("AI:")[-1]
        img_path = self.data[idx]['image'][0] if type(self.data[idx]['image']) is list else self.data[idx]['image']

        print(img_path)
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}
