from transformers import PreTrainedTokenizer, EvalPrediction
from typing import List, Iterable, Optional
from tqdm import tqdm
from collections import Counter, defaultdict
import numpy as np
import re
import string
import collections
import evaluate
import bert_score

from typing import Union, List

def clean_whitespace(text):
        return re.sub(r"\s+", " ", text).strip()

def clean_text(text: str, append_str: str = None):
    text = clean_whitespace(
        re.sub(rf"(/?\w+>|/\w*|<\w+)|[{string.punctuation}]|DOC|PAR|TLE", " ", text)
    )
    if append_str is not None and text:
        # only add '?' if string is not empty otherwise it would never be discared in filtering
        text += append_str
    return text

class MultiLingualEvaluator:
    def __init__(self, lang = "hi", model_name = "bert-base-multilingual-cased"):
        self.lang = lang
        self.model_name = model_name

        self.rouge_metric = evaluate.load("rouge")
        self.bertscore_metric = evaluate.load("bertscore")
        self.bleu_metric = evaluate.load("bleu")

    def compute_semantic(self, 
                         predictions : Union[List[str], str], 
                         references : Union[List[str], str]):
        
        if isinstance(predictions, str):
            predictions = [predictions]

        if isinstance(references, str):
            references = [references]

        if len(predictions) == len(references):
            score = self.bertscore_metric.compute(
                        predictions=predictions,
                        references=references,
                        lang=self.lang,  
                        model_type=self.model_name
                    )
            return {"bertscore" : np.mean(score["f1"])}
        elif len(predictions)>1 and len(references) == 1:
            pass

    def compute_tokenwise(self,
                          predictions : Union[List[str], str],
                          references : Union[List[str], str],
                          strategy : str = "normal"):
 
        if isinstance(predictions, str):
            predictions = [predictions]

        if isinstance(references, str):
            references = [references]

        if strategy == "clean":
            predictions = [clean_text(pred) for pred in predictions]
            references = [clean_text(ref) for ref in references]

        if len(predictions) == len(references):
            score = self.rouge_metric.compute(
                            predictions=predictions,
                            references=references
                        )

            bleu_score = self.bleu_metric.compute(
                                predictions=predictions,
                                references=references
                            )
            
            # print(f"Rouge Scores = {score}", flush = True)

            return {
                "rouge1" : score["rouge1"],
                "rouge2" : score["rouge2"],
                "rougeL" : score["rougeL"],
                "bleu" : bleu_score["bleu"]
            }

            
        