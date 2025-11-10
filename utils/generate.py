import ollama
from typing import List
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import set_seed
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from dataclasses import dataclass
from transformers import BitsAndBytesConfig
import torch
import requests
import re


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
) 

@dataclass
class OModelConfig:
    temperature : float = 0.0
    top_p : float = 0.9
    seed : int = 42

@dataclass
class ReponseBase:
    response : str = ""


class EmbeddingModel_Ollama:
    def __init__(self, model_name = "nomic-embed-text:latest"):
        self.model_name = model_name

    def __call__(self, text : str):
        return ollama.embed(model=self.model_name, input=[text])
    
    def batched_embed(self, texts : List[str]):
        return ollama.embed(model = self.model_name, input=texts)
    

class EmbeddingModel_Huggingface:
    def __init__(self, device = "cpu"):
        self.model_name = "NovaSearch/stella_en_400M_v5"
        self.query_prompt_name = "s2p_query"

        if device.startswith("cuda"):
            print(f"Using Device = {device}")
            self.model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).to(device=device)
        else:
            self.model = SentenceTransformer(
                        "dunzhang/stella_en_400M_v5",
                        trust_remote_code=True,
                        device="cpu",
                        config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}
                    )


    def __repr__(self):
        return f"Embedding model with {self.model_name}"

    def similarity_embed(self, queries : List[str], docs : List[str]):
        query_embeddings = self.model.encode(queries, prompt_name=self.query_prompt_name)
        doc_embeddings = self.model.encode(docs)
        similarities = self.model.similarity(query_embeddings, doc_embeddings)

        return similarities
    
    def get_embedding(self, queries : List[str]):
        query_embeddings = self.model.encode(queries)
        return query_embeddings



class OModel:
    def __init__(self, model_name="llama3.1"):
        self.model_name = model_name

    def __call__(self, prompt):
        return ollama.generate(model=self.model_name, prompt=prompt)
    

class OClientModel:
    def __init__(self, model_name="llama3.1", port = "8080", seed = 42):
        self.model_name = model_name
        self.client = ollama.Client(
        host=f'http://localhost:{port}',
        headers={'x-some-header': 'some-value'}
        )

        self.seed = seed

    def __call__(self, prompt : str, **kwargs):
        if "temperature" in kwargs:
            temperature = float(kwargs["temperature"])
        else:
            temperature = None

        if "top_p" in kwargs:
            top_p = float(kwargs["top_p"])
        else:
            top_p = None

        if "seed" in kwargs:
            self.seed = int(kwargs["seed"])
        

        if top_p is None and temperature is None:
            response = self.client.generate(model=self.model_name, prompt=prompt)
        
        if top_p is None and temperature is not None:
            response = self.client.generate(model = self.model_name, 
                                            prompt=prompt, 
                                            options= {"temperature" : temperature,
                                                      "seed" : self.seed})
            
        if top_p is not None and temperature is None:
            response = self.client.generate(model = self.model_name, 
                                            prompt=prompt, 
                                            options= {"top_p" : top_p,
                                                      "seed": self.seed})
            
        if top_p is not None and temperature is not None:
            response = self.client.generate(model = self.model_name, 
                                            prompt=prompt, 
                                            options= {"top_p" : top_p,
                                                      "temperature" : temperature,
                                                      "seed" : self.seed
                                                    }
                                            )

        if "deepseek" in self.model_name:
            response = re.sub(r"<think>.*?</think>", "", response.response, flags=re.DOTALL).strip()
            response = ReponseBase(response = response)

        return response
    

class HBigModel:
    def __init__(self, model_name = "meta-llama/Meta-Llama-3.1-8B", seed = 21, quantize = False):
        self.model_name = model_name
        set_seed(seed=seed)
        if quantize == False:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map = "auto")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map = "auto", quantization_config=quantization_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast = False)

        self.generation_config = GenerationConfig(max_new_tokens=256, num_return_sequences=3)

    def __call__(self, prompt_, config = None):
        inputs = self.tokenizer(prompt_, return_tensors = "pt").to("cuda:0")
        if config is None:
            output = self.model.generate(**inputs, **self.generation_config.__dict__)
        else:
            output = self.model.generate(**inputs, **config.__dict__)
        output = self.tokenizer.batch_decode(output.sequences, skip_special_tokens = True)   
        
        return output[0]          
    


@dataclass
class GenerationConfig:
    max_new_tokens: int = 8192
    do_sample : bool = True
    temperature: float = 1.0
    top_p: float = 0.9
    return_dict_in_generate : bool = True
    output_scores : bool = True
    num_return_sequences: int = 3
