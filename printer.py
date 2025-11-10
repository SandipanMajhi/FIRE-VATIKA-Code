import datasets

if __name__ == "__main__":
    samples = datasets.load_from_disk("Inference/Lora_2_Inference.hf")
    print(f"Main answer = {samples["answers"][1]}")
    print(f"Model answer = {samples["model_answer"][1]}")
