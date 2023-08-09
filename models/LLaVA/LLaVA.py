from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from llava import LlavaLlamaForCausalLM
from llava.model.builder import load_pretrained_model

# def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto"):
#     kwargs = {"device_map": device_map}
from llava.conversation import conv_templates
from llava import conversation as conversation_lib ## added by me
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from PIL import Image
from ..process import pad_image, resize_image
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

def patch_config(config):
    patch_dict = {
        "use_mm_proj": True,
        "mm_vision_tower": "openai/clip-vit-large-patch14",
        "mm_hidden_size": 1024
    }
    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.')
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)

class KeywordsStoppingCriteria(StoppingCriteria):
                def __init__(self, keywords, tokenizer, input_ids):
                    self.keywords = keywords
                    self.tokenizer = tokenizer
                    self.start_len = None
                    self.input_ids = input_ids

                def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                    if self.start_len is None:
                        self.start_len = self.input_ids.shape[1]
                    else:
                        outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
                        for keyword in self.keywords:
                            if keyword in outputs:
                                return True
                    return False
class LLaVA:
    def __init__(self, model_path, device) -> None:
        print("device ----- ",device)
        model_base = None
        model_name = "llava"
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)
        vision_tower = model.get_vision_tower()
        vision_config = vision_tower.config

        self.image_size = vision_config.image_size  
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.image_processor = image_processor

    def generate(self, image, question, name = 'resize', conv_template = "llava_llama_2"):
        #llava   textVQA none 0.32   pad  0.25   resize 30.4    ct80  none 29.5   pad 63.9    resize  61.5  
        
        qs = question
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        conv = conv_templates[conv_template].copy() ## added by me
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None) ## added by me
    
        prompt = conv.get_prompt()
        
        print("MMOCR conv", conv)
        print("MMOCR prompt", prompt)
        print("MMOCR image size", self.image_size)

        inputs = self.tokenizer([prompt])
        image = Image.open(image)
        
        if name == "pad":
            image = pad_image(image, (self.image_size, self.image_size))
        elif name == "resize":
            image = resize_image(image, (self.image_size, self.image_size))
            
        
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        input_ids = torch.as_tensor(inputs.input_ids).to(self.device)
        if "v0" in conv_template:
            keywords = ['###'] ## added by me
        else:
            keywords = ['</s>'] ## added by me
        
        print("MMOCR keywords", keywords)
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to(self.device),
                do_sample=True,
                temperature=0.2, # 0.9
                max_new_tokens=1024, # 256
                use_cache=True, # added by me
                stopping_criteria=[stopping_criteria])
            
            input_token_len = input_ids.shape[1]
            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

            while True:
                cur_len = len(outputs)
                outputs = outputs.strip()
                for pattern in ['###', 'Assistant:', 'Response:']:
                    if outputs.startswith(pattern):
                        outputs = outputs[len(pattern):].strip()
                if len(outputs) == cur_len:
                    break

            ## added by me
            if conv.sep_style == conversation_lib.SeparatorStyle.TWO or conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
                sep = conv.sep2
            else:
                sep = conv.sep

            try:
                index = outputs.index(sep)
            except ValueError:
                outputs += sep
                index = outputs.index(sep)

            outputs = outputs[:index].strip()

        return outputs