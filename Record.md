# Zero shot Composed Image Retrieval
## Env
```
conda create -n zs_cir python=3.10
conda activate zs_cir
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```
## Note
先训练ke, 然后cot, 最后org

训练的时候不需要用到目标数据集，因为zer-shot

测试的时候CIRR和FashionIQ一起测试

- ft_llm: 微调Phi3
- ft_qwen: 微调qwen


之前的记录：
test: pip install transformers==4.44.2
train:pip install transformers==4.41.2
## Sigularity Problem
RuntimeError: Failed to import transformers.trainer because of the following error (look up to see its traceback):
cannot import name 'is_mlu_available' from 'accelerate.utils'