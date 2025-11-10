import datasets
from tqdm import tqdm
from utils.multilingual_evaluator import MultiLingualEvaluator


if __name__ == "__main__":
    evaluator = MultiLingualEvaluator()
    vatika_dev_set = datasets.load_from_disk("Inference/Lora_1_testv1_Inference.hf")

    print(f"Rouge scores = {evaluator.compute_tokenwise(predictions = vatika_dev_set["model_answer"], 
                                                        references=vatika_dev_set["answers"])}")

    print(f"BERT Score = {evaluator.compute_semantic(predictions = vatika_dev_set["model_answer"],
                                                     references=vatika_dev_set["answers"])}")