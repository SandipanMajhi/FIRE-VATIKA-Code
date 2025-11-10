from unsloth import FastLanguageModel
from transformers import TextStreamer
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from tqdm import tqdm
import torch
import re
import datasets
import argparse
from collections import defaultdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="argparse for coding the inference module.")
    parser.add_argument("--data", type = str, help = "Data path for inference QA", default = "Data/vatika-validation.hf")
    parser.add_argument("--aug", type = str, help = "VATIKA Augmented dataset with inference")
    parser.add_argument("--model_path", type = str, help = "Model path", default = "lora_model")

    args = parser.parse_args()

    dev_set_path = args.data
    dev_data = datasets.load_from_disk(dev_set_path)

    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_path, # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    text_streamer = TextStreamer(tokenizer)

    custom_prompt = """यहाँ दिए गए संदर्भ के आधार पर निम्नलिखित प्रश्न का उत्तर दें। आपका उत्तर सार्थक और व्याकरणिक रूप से सही होना चाहिए।
संदर्भ:{}
प्रश्न:{}
उत्तर:{}"""

    EOS_TOKEN = tokenizer.eos_token
    max_new_tokens = 512

    augmented_samples = defaultdict(list)

    for idx in tqdm(range(len(dev_data))):
        context = dev_data["contexts"][idx]
        question = dev_data["questions"][idx]
        gold_answer = dev_data["answers"][idx]

        # print(f"Gold Answers = {gold_answer}", flush = True)

        inputs = tokenizer([
                    custom_prompt.format(
                        context,
                        question,
                        ""
                    )
                    
                ], return_tensors = "pt").to("cuda")

        input_len = inputs.input_ids.shape[1]

        outputs = model.generate(**inputs, max_new_tokens = max_new_tokens, top_p = 0.9, do_sample = True)

        new_tokens = outputs[0, input_len:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        # print(f"Generated_answers = {generated_text}", flush = True)
        # print(f"Parsed answer = {parse(text = generated_text)}")
        # _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = max_new_tokens)

        augmented_samples["contexts"].append(context)
        augmented_samples["questions"].append(question)
        augmented_samples["answers"].append(gold_answer)
        augmented_samples["model_answer"].append(generated_text)

        if idx % 10 == 0:
            saved_samples = datasets.Dataset.from_dict(augmented_samples)
            saved_samples.save_to_disk(args.aug)

    
    saved_samples = datasets.Dataset.from_dict(augmented_samples)
    saved_samples.save_to_disk(args.aug)



