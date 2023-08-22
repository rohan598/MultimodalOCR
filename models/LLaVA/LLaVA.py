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

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        patch_config(model_path)
        model = LlavaLlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
        self.image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device = device, dtype=torch.float16)
        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        self.image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # model_base = None
        # model_name = "llava"
        # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)
        # # print(model.config)
        # mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        # # tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        # if mm_use_im_start_end:
        #     print("in here")
        #     tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        # vision_tower = model.get_vision_tower()
        # vision_tower.to(device = device, dtype=torch.float16)
        # vision_config = vision_tower.config
        # # print(vision_config)
        # # vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        # vision_config.use_im_start_end = mm_use_im_start_end
        # if mm_use_im_start_end:
        #     print("in here 2")
        #     vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        # # self.image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
        # self.image_size = vision_config.image_size  
        # self.model = model
        # self.tokenizer = tokenizer
        # self.device = device
        # self.image_processor = image_processor

    def generate(self, image, question, name = 'resize', conv_template = "llava_llama_2", qs_template=1, temperature = 0.2):
        #llava   textVQA none 0.32   pad  0.25   resize 30.4    ct80  none 29.5   pad 63.9    resize  61.5  
        qs = question + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len + DEFAULT_IM_END_TOKEN
        # qs = question
        # if self.model.config.mm_use_im_start_end:
        #     if qs_template == 1:
        #         qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        #     else:
        #         qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        # else:
        #     if qs_template == 1:
        #         qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        #     else:
        #         qs = qs + '\n' + DEFAULT_IMAGE_TOKEN

        conv = conv_templates[conv_template].copy() ## added by me
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None) ## added by me
    
        prompt = conv.get_prompt()
        
        # print("MMOCR conv", conv)
        # print("MMOCR prompt", prompt)
        # print("MMOCR image size", self.image_size)

        inputs = self.tokenizer([prompt])
        image = Image.open(image)
        
        if name == "pad":
            image = pad_image(image, (336, 336))
        elif name == "resize":
            image = resize_image(image, (336, 336))
            
        
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
                temperature=0.9,
                max_new_tokens=256, # 256
                # use_cache=True, # added by me
                stopping_criteria=[stopping_criteria])
            
            input_token_len = input_ids.shape[1]
            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

            while True:
                cur_len = len(outputs)
                outputs = outputs.strip()
                print(outputs)
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