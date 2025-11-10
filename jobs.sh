######################### How to run all codes ######################

##### build train data from json data #####
nohup python build_dataset.py \
    --data_path Data/validation.json \
    --output_path Data/vatika-validation.hf \
    > build_data.log &


##### Build Synthetic Data Ollama ######

#### Ollama #### 
#### Assumes Ollama serve is running in the background and the user knows the port ####
#### ollama pull <model_name> ####

nohup python main_synthetic_data.py \
    --data_path Data/vatika-train.hf \
    --ollama_model_name phi4:14b \
    --ollama_port 11434 \
    --num_tries 5 \
    --output_path Data/synth-phi-4.hf \
    > synthetic_data_phi_4.log &


##### Finetuning code #####
CUDA_VISIBLE_DEVICES=0 nohup python unsloth_trainer.py \
    --model_name meta-llama/Llama-3.1-8B \
    --train_set_path Data/vatika-train.hf \
    --output_dir Outputs/llama3_train \
    --model_save_path Models/llama3_train \
    --batch_size 8 \
    --num_epochs 3 \
    > training.log &


##### Inference Code #####
CUDA_VISIBLE_DEVICES=0 nohup python unsloth_inference.py \
    --data Data/vatika-validation.hf \
    --aug Inferences/vatika-validation-inference.hf \
    --model_path Models/llama3_train \
    > inference.log &








