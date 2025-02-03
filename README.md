# Djuplet
Experiment to try to build a Norwegian R1-like model, experimenting on a synhtetic data task.

This repo has the following steps:
* Download data from Wikipedia to create a set of clean text paragraphs. Add a distortion task to this dataset. From this the following datasets are created
- [Wiki Paragraphs Norwegian](https://huggingface.co/datasets/pere/wiki_paragraphs_norwegian)
- [Wiki Paragraphs English](https://huggingface.co/datasets/pere/wiki_paragraphs_english)

* Use the Wikipedi Paragraphs dataset for get DeepSeek to generate reasoning responses. Create a dataseet and upload. This creates the following dataset:
- [Norwegian Reasoning](https://huggingface.co/datasets/pere/norwegian_reasoning)


* Apply SFT-training on a Llama Base model to get it to accept reasoning. (ongoing) 

* Reinforcement learning on the Wiki Paragraphs datasets. (pending)


## Requirements

Ensure you have the following Python packages installed. You can install them using the provided `requirements.txt` file and that your DeepSeek-API-key is set:

```bash
pip install -r requirements.txt
export DeepSeekApi=<your_deepseek_api_key>
```