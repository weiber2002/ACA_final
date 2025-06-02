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
