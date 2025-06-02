# Speculative Decoding

Execute the simple speculative decoding by
```
python infer.py --device cuda:0 
or 
python infer.py --device cuda:1
```
You can adjust what you want like gamma, temperature, output length in the `infer.py` file.
You can select what you want to test, like different models, and speculative decoding, cascaded speculative decoding, and so on.

Remind that you need to have interactive mode, enable line 56

# Cache Speculative Decoding


Execute the cache speculative decoding by
```
python cache_infer.py 
```
If you want inferece on NCHC Nano25, you can directly use the script file. To be noticed, I need to add you in the project first so you can run the script file.
```
sbatch cache_infer.sh
```

You can adjust what you want like gamma, temperature, output length in the `cache_infer.py` file.
You can change the model as your GPU can afford. (on NCHC can run "Qwen/Qwen3-32B-FP8").
You can change the input prompt on line 345. Or you can use line 339~ 345 for interactive version like Speculative Decoding.
This version is only for use KV cache, so make sure self.cache = True
You can set self.spec, self.target_gen, self.dr  to open the speculative decoding, only target and only drafter. We recommand to set them True to compare the throughput improvement.

# Tensor Parallelism

Enter the TP dir. first
```
cd TP
```
If you have multi-GPU enviroment, you can directly run

```
torchrun  --nproc_per_node=1 TP.py
```
change the nproc_per_node from 1, 2, 4,8 can change the GPU NUM

If you want run on NCHC Nano25, you can directly use the script file. To be noticed, I need to add you in the project first so you can run the script file.

```
sbatch TP.sh
```
You need to change the sh file #SBATCH --gpus-per-node=  to set GPU NUM you want use and corresponding --nproc_per_node= . IF you face port if full, you can add this command in script file
```
$CONDA_PREFIX/bin/conda run -n "$CONDA_DEFAULT_ENV" --master-port=29505 torchrun  --nproc_per_node=4 TP.py
```

USE_TEMPERATURE_SAMPLING  is to determined whether sample with temparature (open can get more reasonable response)
my_max_new_tokens is to set  max new tokens
line 868 can change your input prompt
You will find all GPU setting genertate same output tokens.


## How to use

### 0. Installation
This project requires Python 3.7 or later and the following dependencies:

```
rich
tqdm
termcolor
tokenizers>=0.19.1
torch>=2.3.0
transformers>=4.41.1
accelerate>=0.30.1
bitsandbytes>=0.43.1
```
