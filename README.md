
# Paraphrase Generation with Pegasus

Welcome to the Paraphrase Generation with Pegasus project! This project focuses on generating paraphrases using the Pegasus model.

## Introduction

Paraphrase generation involves creating different versions of a given sentence with the same meaning. In this project, we leverage the power of Pegasus to generate paraphrases on the Quora Question Pairs dataset.

## Dataset

For this project, we will use the Quora Question Pairs dataset. You can create your own dataset and place it in the `data/quora_question_pairs.csv` file.

## Project Overview

### Prerequisites

- Python 3.6 or higher
- PyTorch
- Hugging Face Transformers
- Datasets

### Installation

To set up the project, follow these steps:

```bash
# Clone this repository and navigate to the project directory:
git clone https://github.com/your-username/pegasus_paraphrase_generation.git
cd pegasus_paraphrase_generation

# Install the required packages:
pip install -r requirements.txt

# Ensure your data includes paraphrase pairs. Place these files in the data/ directory.
# The data should be in a CSV file with two columns: question1 and question2.

# To fine-tune the Pegasus model for paraphrase generation, run the following command:
python scripts/train.py --data_path data/quora_question_pairs.csv

# To evaluate the performance of the fine-tuned model, run:
python scripts/evaluate.py --model_path models/ --data_path data/quora_question_pairs.csv
