import unsloth
from unsloth import FastLanguageModel
from transformers import TextStreamer
from unsloth import is_bfloat16_supported
from tqdm import tqdm
import torch
import re
import datasets
import random
from collections import defaultdict
from typing import List
from tqdm import tqdm
from utils.generate import OClientModel, OModelConfig

from transformers import set_seed

class SynthData:
    def __init__(self, 
                 model_name : str, 
                 max_seq_len : int,
                 num_tries : int,
                 max_new_tokens : int,
                 output_path : str):
        
        self.num_tries = num_tries
        self.max_new_tokens = max_new_tokens
        self.output_path = output_path

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name, # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length = max_seq_len,
            dtype = None,
            load_in_4bit = True,
        )

        FastLanguageModel.for_inference(self.model)

    
    def parse(self, text : str):
        pattern = re.compile(
            r'प्रश्न:\s*(.*?)\s*उत्तर:\s*(.*?)(?=\n*प्रश्न:|$)', 
            re.DOTALL
        )
        qa_pairs = pattern.findall(text)
        cleaned_pairs = [(q.strip(), a.strip()) for q, a in qa_pairs]
        return cleaned_pairs
    
    @classmethod
    def prepare_data_ollama(cls, 
                            ollama_model : OClientModel, 
                            contexts : List[str], 
                            num_tries : int,
                            output_path : str):

        def parse(text : str):
            pattern = re.compile(
                r'प्रश्न:\s*(.*?)\s*उत्तर:\s*(.*?)(?=\n*प्रश्न:|$)', 
                re.DOTALL  # Allows '.' to match newlines, handling multi-line Q/A
            )
            qa_pairs = pattern.findall(text)
            cleaned_pairs = [(q.strip(), a.strip()) for q, a in qa_pairs]
            return cleaned_pairs
        

        prompt_ = """निम्नलिखित संदर्भ को देखते हुए, उसमें से प्रश्न-उत्तर जोड़े आउटपुट करें। आपका आउटपुट निम्नलिखित प्रारूप में होना चाहिए।

प्रश्न: <प्रश्न>
उत्तर: <उत्तर>

प्रश्न: <प्रश्न>
उत्तर: <उत्तर>

केवल प्रश्न-उत्तर जोड़े आउटपुट करें और कुछ नहीं।

संदर्भ:{}"""

        augmented_samples = defaultdict(list)

        for idx in tqdm(range(len(contexts))):
            input_prompt = prompt_.format(contexts[idx])

            for _ in range(num_tries):
                config = OModelConfig(temperature=0.7, top_p=0.9, seed = random.randint(0,1000000001))
                generated_qa_pairs = ollama_model(input_prompt, **config.__dict__).response
                generated_qa_pairs = parse(generated_qa_pairs)

                generated_qa_pairs = list(set(generated_qa_pairs))

                for q,a in generated_qa_pairs:
                    augmented_samples["contexts"].append(contexts[idx])
                    augmented_samples["questions"].append(q)
                    augmented_samples["answers"].append(a)

            if idx % 10 == 0:
                saved_samples = datasets.Dataset.from_dict(augmented_samples)
                saved_samples.save_to_disk(output_path)

        saved_samples = datasets.Dataset.from_dict(augmented_samples)
        saved_samples.save_to_disk(output_path)

    
    def prepare_data(self, contexts : List[str]):

        prompt_ = """निम्नलिखित संदर्भ को देखते हुए, उसमें से प्रश्न-उत्तर जोड़े आउटपुट करें। आपका आउटपुट निम्नलिखित प्रारूप में होना चाहिए।

प्रश्न: <प्रश्न>
उत्तर: <उत्तर>

प्रश्न: <प्रश्न>
उत्तर: <उत्तर>

केवल प्रश्न-उत्तर जोड़े आउटपुट करें और कुछ नहीं।

संदर्भ:{}"""

        augmented_samples = defaultdict(list)

        for idx in tqdm(range(len(contexts))):
            inputs = self.tokenizer([
                    prompt_.format(
                        contexts[idx]
                    )
                    
                ], return_tensors = "pt").to("cuda")

            input_len = inputs.input_ids.shape[1]

            for _ in range(self.num_tries):

                set_seed(random.randint(0, 100000001))
                outputs = self.model.generate(**inputs, 
                                              max_new_tokens = self.max_new_tokens, 
                                              top_p = 0.9, 
                                              temperature = 0.7,
                                              do_sample = True)

                new_tokens = outputs[0, input_len:]

                generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

                print(f"Generated Text = {generated_text}", flush = True)

                generated_qa_pairs = self.parse(generated_text)
                generated_qa_pairs = list(set(generated_qa_pairs))

                for q,a in generated_qa_pairs:
                    augmented_samples["contexts"].append(contexts[idx])
                    augmented_samples["questions"].append(q)
                    augmented_samples["answers"].append(a)


            if idx % 10 == 0:
                saved_samples = datasets.Dataset.from_dict(augmented_samples)
                saved_samples.save_to_disk(self.output_path)

        saved_samples = datasets.Dataset.from_dict(augmented_samples)
        saved_samples.save_to_disk(self.output_path)

        


            

