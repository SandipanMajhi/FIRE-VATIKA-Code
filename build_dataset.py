import datasets
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

from utils.generate import OClientModel


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type = str, help = "Data path")
    parser.add_argument("--output_path", type = str, help = "Output Path")

    args = parser.parse_args()

    # train_file_path = "VATIKA/DATASET/train.json"
    # dev_file_path = "VATIKA/DATASET/validation.json"
    # test_file_path = "Data/test.json"

    with open(args.data_path, "r") as fp:
        samples = json.load(fp)

    vatika_contexts = []
    questions = []
    answers = []
    ids = []

    augmented_samples = defaultdict(list)

    for i in tqdm(range(len(samples["domains"])), disable=False, desc = "In Domain:"):
        contexts = samples["domains"][i]["contexts"]
        for j in range(len(contexts)):
            context = contexts[j]["context"]
            qas = contexts[j]["qas"]

            for qa in qas:
                vatika_contexts.append(context)
                questions.append(qa["question"])
                answers.append(qa["answer"])
                ids.append(qa["id"])

    augmented_samples["contexts"] = vatika_contexts
    augmented_samples["questions"] = questions
    augmented_samples["answers"] = answers
    augmented_samples["ids"] = ids

    augmented_samples = datasets.Dataset.from_dict(augmented_samples)
    augmented_samples.save_to_disk(args.output_path)

    

    






    