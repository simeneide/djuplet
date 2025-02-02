---
license: cc-by-sa-3.0
task_categories:
- text-generation
- text-classification
language:
- no
pretty_name: WIKI Paragraphs Norwegian
configs:
- config_name: default
  data_files:
    - split: train
      path: train.jsonl
    - split: validation
      path: validation.jsonl
    - split: test
      path: test.jsonl
    - split: validation1000
      path: validation1000.jsonl
    - split: test1000
      path: test1000.jsonl
    - split: validation100
      path: validation100.jsonl
    - split: test100
      path: test100.jsonl
    - split: pretrain
      path: pretrain.jsonl
    - split: reserve
      path: reserve.jsonl
version: 1.0.0
citation: >
  This dataset contains content from Wikipedia under CC BY-SA 3.0 license.
dataset_info:
  features:
    - name: text
      dtype: string
    - name: url
      dtype: string
    - name: paragraph_number
      dtype: int64
    - name: corrupt
      dtype: string
    - name: corrupt_level
      dtype: int64

  splits:
  - name: train
    num_examples: 1000000
  - name: validation
    num_examples: 10000
  - name: test
    num_examples: 10000
  - name: validation1000
    num_examples: 1000
  - name: test1000
    num_examples: 1000
  - name: validation100
    num_examples: 100
  - name: test100
    num_examples: 100
  - name: pretrain
    num_examples: 10000
  - name: reserve
    num_examples: 100000
---
# WIKI Paragraphs Norwegian

A multi-split dataset for machine learning research and evaluation, containing text samples in JSON Lines format.

## Features
- **Multiple splits** for different use cases
- **Random shuffle** with Fisher-Yates algorithm
- **Structured format** with text and metadata
- **Size-varied validation/test sets** (100 to 10k samples)

## Splits Overview
| Split Name          | Samples | Typical Usage          |
|---------------------|--------:|------------------------|
| `train`             | 1,000,000 | Primary training data  |
| `validation`        |   10,000 | Standard validation    |
| `test`              |   10,000 | Final evaluation       |
| `validation1000`    |    1,000 | Quick validation       |
| `test1000`          |    1,000 | Rapid testing          |
| `validation100`     |      100 | Debugging/development  |
| `test100`           |      100 | Small-scale checks     |
| `pretrain`          |   10,000 | Pre-training phase     |
| `reverse`           |  100,000 | Special tasks          |

**Total Samples:** 1,132,200

## License
**Creative Commons Attribution-ShareAlike 3.0**  
[![CC BY-SA 3.0](https://licensebuttons.net/l/by-sa/3.0/88x31.png)](https://creativecommons.org/licenses/by-sa/3.0/)

This dataset inherits Wikipedia's licensing terms:
- **Attribution Required**  
- **ShareAlike Mandatory**  
- **Commercial Use Allowed**

## Usage
```python
from datasets import load_dataset

# Load main training split
dataset = load_dataset("your-username/dataset-name", split="train")

# Access smaller validation split
val_100 = load_dataset("your-username/dataset-name", "validation100")
```

## Data Structure

Each line contains JSON:

```json
Copy
{
  "text": "Full text content...",
  "metadata": {
    "source": "Wikipedia", 
    "timestamp": "2023-01-01",
    "url": "https://..."
  }
}
```

## Notes

All splits accessible via:
load_dataset(repo_id, split_name)
Non-standard splits (e.g., reverse) require explicit config:
split="reverse"
When using, include attribution:
"Contains content from Wikipedia under CC BY-SA 3.0"
