[Paper](https://arxiv.org/pdf/2305.07895.pdf). The online evaluation pipeline is scheduled to release. 

# Results

Results are available in answer_save folder. It should be noted that for BLIP2OPT, when using the inference code on Hugging Face, the accuracy of text recognition is high, but the model outputs nothing for the VQA tasks. Conversely, when using the LAVIS library for inference, the accuracy of text recognition is low, while the VQA accuracy is normal. We believe that the inference process of BLIP2OPT still needs to be optimized. In our experiments, we take the maximum value of the two methods as the final result.

![image](https://github.com/echo840/MultimodalOCR/assets/87795401/523e0421-7eca-4d15-89f1-3f7348321055)

Visualization results

![rvk](https://github.com/echo840/MultimodalOCR/assets/87795401/21982aba-d063-4a52-a045-8d16e0e98f71)


# Data Download
| Data file | Size |
| --- | ---: |
|[text recognition](https://pan.baidu.com/s/1Ba950d94u8RQmtqvkLBk-A) code:iwyn | 1.37GB |
|[STVQA](https://rrc.cvc.uab.es/?ch=11&com=downloads) End-to-End Task-3 and Training images|1.88GB|
|[ocrVQA](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_)|--|
|[textVQA](https://textvqa.org/dataset/) val set|6.6GB|
|[docVQA](https://rrc.cvc.uab.es/?ch=17&com=downloads) Task 1 Validation set|0.8GB|
|[ESTVQA](https://cloudstor.aarnet.edu.au/plus/s/LSishuuSE5DBKJp)|5.2GB|
|[SROIE](https://rrc.cvc.uab.es/?ch=13&com=downloads) Task 3 test set|0.19GB|
|[FUNSD](https://guillaumejaume.github.io/FUNSD/download/)|16MB|
|[POIE](https://drive.google.com/file/d/1eEMNiVeLlD-b08XW_GfAGfPmmII-GDYs/view)|0.43GB|
|[HME100K](https://ai.100tal.com/openData/formulaRecognition)|0.69GB|

TextVQA, KIE and HME will be updated soon.

We assume that your symlinked `data` directory has the following structure:

```
data
|_ IC13_857
|_ IC15_1811
|_ ...
|_ ESTVQA
|_ textVQA
|_ ...
|_ FUNSD
|_ POIE
```


# Usage

eval on all datasets
```Shell
python eval.py --model_name LLaVA --eval_all
```

eval on one dataset
```Shell
python eval.py --model_name LLaVA --eval_textVQA
```
```Shell
python eval.py --model_name LLaVA --eval_ocr --ocr_dataset_name "ct80 IIIT5K"
```
The results will be saved at answer folder.

If you want to add a new model, please write its inference function under the folder "models", and update the get_model function in eval.py. An example inference code is as follows：

```Shell
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from ..process import pad_image, resize_image
class lavis:
    def __init__(self, model_name, model_type, device) -> None:
        model, vis_processors, txt_processors = load_model_and_preprocess(name = model_name, model_type = model_type, is_eval=True, device=device)
        self.model_name = model_name
        self.model = model
        self.vis_processors = vis_processors
        self.txt_processors = txt_processors
        self.device = device
    def generate(self, image, question, name='resize'):
        if 'opt' in self.model_name:
            prompt = f'Question: {question} Answer:'
        elif 't5' in self.model_name:
            prompt = f'Question: {question} Short answer:'
        else:
            prompt = f'Question: {question} Answer:'
        image = Image.open(image).convert("RGB")
        if name == "pad":
            image = pad_image(image, (224,224))
        elif name == "resize":
            image = resize_image(image, (224,224))
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        prompt = self.txt_processors["eval"](prompt)
        answer = self.model.predict_answers(samples={"image": image, "text_input": prompt}, inference_method="generate", max_len=48, min_len=1)[0]
        return answer
```

# Related Projects
- [LLaVA](https://github.com/haotian-liu/LLaVA.git)
- [MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4.git)
- [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl.git)
- [OpenFlamingo](https://github.com/mlfoundations/open_flamingo.git)
- [LAVIS](https://github.com/salesforce/LAVIS.git)
