import datasets
import argparse
from tqdm import tqdm
import pickle as pkl

from utils.generate import OClientModel, OModelConfig
from collections import defaultdict

from typing import List, Dict




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="argparse for coding the inference module.")
    parser.add_argument("--data", type = str, help = "Data path for inference QA")
    parser.add_argument("--start_id", type = int, help = "start index")
    parser.add_argument("--end_id", type = int, help = "end index")
    parser.add_argument("--port", type = str, help = "port of the model")
    parser.add_argument("--aug", type = str, help = "VATIKA Augmented dataset")

    args = parser.parse_args()

    model = OClientModel(port = args.port)
    samples = datasets.load_from_disk(args.data)
    start_id = int(args.start_id)
    end_id = min(int(args.end_id), len(samples))

    # instructions = "यहाँ दिए गए संदर्भ के आधार पर निम्नलिखित प्रश्न का उत्तर दें। केवल अपना उत्तर दें, और कुछ भी नहीं।"
    instructions = "यहाँ दिए गए संदर्भ के आधार पर निम्नलिखित प्रश्न का उत्तर दें। आपका उत्तर सार्थक और व्याकरणिक रूप से सही होना चाहिए।"

    augmented_samples = defaultdict(list)

    for idx in tqdm(range(start_id, end_id), desc = "In Samples Iterate:", disable = False):
        context = samples["contexts"][idx]
        question = samples["questions"][idx]
        gold_answers = samples["answers"][idx]

        prompt = f"""{instructions}
संदर्भ:{context}
प्रश्न:{question}
उत्तर:"""
        generation_config = OModelConfig()
        output = model(prompt = prompt, **generation_config.__dict__).response

        augmented_samples["contexts"].append(context)
        augmented_samples["questions"].append(question)
        augmented_samples["gold_answers"].append(gold_answers)
        augmented_samples["model_answer"].append(output)

        if idx % 10 == 0:
            saved_samples = datasets.Dataset.from_dict(augmented_samples)
            saved_samples.save_to_disk(args.aug)

    saved_samples = datasets.Dataset.from_dict(augmented_samples)
    saved_samples.save_to_disk(args.aug)







