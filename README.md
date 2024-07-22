To reproduce our experiments, first install all dependency in requirements.txt. Then download `chatalpaca-10k.json` from https://github.com/cascip/ChatAlpaca.

Next set your huggingface token to your environment variable with the following command. It is used to download Llama-2 checkpoint from huggingface (you need to request access first).

`export HFTOKEN=$YourToken`

Run evaluation.py with the following arguments.

`python evaluation.py --approx_model_name $small_model_name --target_model_name $large_model_name --max_tokens 128 --max_seconds 1000 --dataset $dataset`

the model names used in our experiments are: `JackFram/llama-68m` and `meta-llama/Llama-2-13b-hf`, `facebook/opt-125m` and `facebook/opt-13b` 

The dataset argument can be `chatalpaca`, `ChatGPT`, and `cnndm`.



