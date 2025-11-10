import argparse
import datasets
from tqdm import tqdm
from datasets import load_from_disk

from utils.generate import OClientModel, OModelConfig
from utils.synthetic_data import SynthData



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type = str, help  = "Main data path")
    parser.add_argument("--ollama_model_name", type = str, help = "Model name")
    parser.add_argument("--ollama_port", type = str, help = "11434")
    parser.add_argument("--num_tries", type = int, help = "Num tries")
    parser.add_argument("--output_path", type = str, help = "dataset save path")
    

    args = parser.parse_args()

    samples = load_from_disk(args.data_path)

    unique_contexts = []
    for idx in tqdm(range(len(samples))):
        context = samples["contexts"][idx]
        if context not in unique_contexts:
            unique_contexts.append(context)

    model = OClientModel(model_name = args.ollama_model_name, port = args.ollama_port)
    SynthData.prepare_data_ollama(ollama_model = model, 
                                contexts=unique_contexts, 
                                num_tries=int(args.num_tries), 
                                output_path=args.output_path)
    


    
