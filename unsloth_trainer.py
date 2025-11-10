from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import torch
import datasets
import argparse




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type = str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--train_set_path", type = str, help = "Train data")
    parser.add_argument("--output_dir", type = str, help = 'Output Path')
    parser.add_argument("--model_save_path", type = str, help = "Model Save Path")
    parser.add_argument("--batch_size", type = int, help = "Per device batch size")
    parser.add_argument("--num_epochs", type = int, help = "Num Epochs")


    args = parser.parse_args()

    # train_set_path = "Data/Vatika_train.hf"
    # dev_set_path = "Data/Vatika_dev.hf"

    train_data = datasets.load_from_disk(args.train_set_path)
    # dev_data = datasets.load_from_disk(dev_set_path)

    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    fourbit_models = [
        "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
        "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
        "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
        "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
        "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
        "unsloth/Phi-3-medium-4k-instruct",
        "unsloth/gemma-2-9b-bnb-4bit",
        "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
    ] # More models at https://huggingface.co/unsloth

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, 
        bias = "none",    
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,  
        loftq_config = None,
    )

    custom_prompt = """यहाँ दिए गए संदर्भ के आधार पर निम्नलिखित प्रश्न का उत्तर दें। आपका उत्तर सार्थक और व्याकरणिक रूप से सही होना चाहिए।
संदर्भ:{}
प्रश्न:{}
उत्तर:{}"""

    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    def formatting_prompts_func(examples):
        contexts = examples["contexts"]
        questions = examples["questions"]
        answers = examples["answers"]
        texts = []
        for context, question, answer in zip(contexts, questions, answers):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            if len(context) > 0 and len(question) > 0 and len(answer) > 0:
                text = custom_prompt.format(context, question, answer) + EOS_TOKEN
                # print(text, flush = True)
                texts.append(text)
        return { "text" : texts, }
    pass

    train_dataset = train_data.map(formatting_prompts_func, batched=True)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = int(args.batch_size),
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs=int(args.num_epochs),
            # num_train_epochs = 1, # Set this for 1 full training run.
            # max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none",
    ))

    trainer_stats = trainer.train()

    model.save_pretrained(f"{args.model_save_path}")  # Local saving
    tokenizer.save_pretrained(f"{args.model_save_path}")