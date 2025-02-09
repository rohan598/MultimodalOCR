from torch.utils.data import Dataset
import os
import json
class textVQADataset(Dataset):
    def __init__(
        self,
        image_dir_path= "./data/textVQA/train_images",
        ann_path= "./data/textVQA/TextVQA_0.5.1_val.json"
    ):
        
        self.data = json.load(open(ann_path, "r"))["data"]
        self.image_dir_path = image_dir_path
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['question']
        answers = self.data[idx]['answers']
        img_path = os.path.join(self.image_dir_path, f"{self.data[idx]['image_id']}.jpg")
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}

class docVQADataset(Dataset):
    def __init__(
        self,
        image_dir_path= "./data/docVQA/val",
        ann_path= "./data/docVQA/val/val_v1.0.json",
    ):
        
        self.data = json.load(open(ann_path, "r"))["data"]
        self.image_dir_path = image_dir_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['question']
        answers = self.data[idx]['answers']
        img_path = os.path.join(self.image_dir_path, self.data[idx]['image'])
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}

class ocrVQADataset(Dataset):
    def __init__(
        self,
        image_dir_path= "./data/ocrVQA/images",
        ann_path= "./data/ocrVQA/dataset.json",
    ):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        dataset = json.load(open(ann_path, "r"))
        for idx, data in enumerate(dataset):
            questions =  dataset[data]['questions']
            for index, question in enumerate(questions):
                image_file = os.path.join(image_dir_path, f'{data}.jpg')
                gt_answers = dataset[data]['answers'][index]
                self.image_list.append(image_file)
                self.answer_list.append(gt_answers)
                self.question_list.append(question)
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}

class STVQADataset(Dataset):
    def __init__(
        self,
        image_dir_path= "./data/STVQA",
        ann_path= "./data/STVQA/train_task_3.json",
    ):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        data = json.load(open(ann_path, "r"))
        for i in range(len(data['data'])):
            image_path = image_dir_path+'/'+data['data'][i]['dataset']+'/'+data['data'][i]['file_name']
            self.image_list.append(image_path)
            self.answer_list.append(data['data'][i]['answers'])
            self.question_list.append(data['data'][i]['question'])
            if i+1==5000:
                break
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}
        
class ESTVQADataset(Dataset):
    def __init__(
        self,
        image_dir_path= "./data/ESTVQA/images/train",
        ann_path= "./data/ESTVQA/annotations/train.json",
    ):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        with open(ann_path,'r') as f:
            data = json.load(f)
            for i in range(len(data)):
                image_path = os.path.join(image_dir_path, data[i]['image'])
                for j in range(len(data[i]['annotation'])):
                    question = data[i]['annotation'][j]['question']
                    answer = data[i]['annotation'][j]['answer']
                    self.image_list.append(image_path)
                    self.question_list.append(question)
                    self.answer_list.append(answer)
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}
    
class ChartQADataset(Dataset):
    def __init__(
        self,
        image_dir_path= "./test/png",
        ann_path= "./test/test_human.json",
    ):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        with open(ann_path,'r') as f:
            data = json.load(f)
            for i in range(len(data)):
                image_path = os.path.join(image_dir_path, data[i]['imgname'])
                self.image_list.append(image_path)
                self.question_list.append(data[i]["query"])
                self.answer_list.append(data[i]["label"])
                    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}
    
class InfoVQADataset(Dataset):
    def __init__(
        self,
        image_dir_path= "./infographicsvqa/infographicsvqa_images",
        ann_path= "./infographicsvqa/infographicsvqa_qas/infographicsVQA_val_v1.0_withQT",
    ):
        
        self.data = json.load(open(ann_path, "r"))["data"]
        self.image_dir_path = image_dir_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['question']
        answers = self.data[idx]['answers']
        img_path = os.path.join(self.image_dir_path, self.data[idx]['image_local_name'])
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}