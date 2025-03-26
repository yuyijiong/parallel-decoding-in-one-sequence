# Parallel Decoding wihtin One Sequence

To address the inefficiency of long reasoning, we leverage the inherent parallelizability of certain tasks to accelerate the reasoning process. 
Specifically, when multiple parallel reasoning branches exist, 
we decode multiple tokens per step using a specialized attention mask, processing them within a single sequence.

![method](./method.png)


## Run experiment

The experiment will run the same dataset with normal decoding and our parallel decoding method respectively, 
and compare the output and decoding speed of them.
The results will be saved in the `results` folder.

### Use Normal attention implementation

environment:
``
transformers>=4.49.0
torch>=2.5.0
``

To run our method with Qwen2 (or 2.5) models on the dataset of retrieval task
```bash
python run.py --model_path <model_path> --task "retrieval"
```

To run our method with Qwen2 (or 2.5) models on the dataset of multi-document QA
```bash
python run.py --model_path <model_path> --task "multi-document-qa"
```
### Use flash-attention-2

First, install the flash-attention-2 package of our modified version
```bash
cd flash-attention
python setup.py install
```

To run our method with Qwen2 (or 2.5) models on the dataset of retrieval task
```bash
python run.py --model_path <model_path> --task "retrieval" --attn_implementation "flash_attention_2"
```

To run our method with Qwen2 (or 2.5) models on the dataset of multi-document QA
```bash
python run.py --model_path <model_path> --task "multi-document-qa" --attn_implementation "flash_attention_2"
```