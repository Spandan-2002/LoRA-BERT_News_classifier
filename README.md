
# 🧠 LoRA-BERT_News_Classifier

A parameter-efficient fine-tuning (PEFT) project applying LoRA to a RoBERTa-base model for classifying news headlines using the AG News dataset. Built for Deep Learning Project 2 by **Team SPAR**.

## 🚀 Project Overview

Fine-tuning large language models like RoBERTa can be computationally expensive. This project demonstrates how **Low-Rank Adaptation (LoRA)** enables efficient fine-tuning of transformer models by updating only a subset of parameters. We achieve competitive accuracy on a 4-class news classification task with minimal computational overhead.

📌 **Project link:** [LoRA-BERT_News_classifier](https://github.com/Spandan-2002/LoRA-BERT_News_classifier)

## 📂 Dataset

- **Source:** [AG News](https://huggingface.co/datasets/ag_news)
- **Split:** 120K training samples, 7.6K test samples
- **Classes:** `World`, `Sports`, `Business`, `Sci/Tech`

## 🛠️ Methodology

### Model
- Base model: `roberta-base`
- Head: `RobertaForSequenceClassification`
- Tokenizer: `RobertaTokenizer`
- Task type: Sequence Classification
- PEFT using `peft.LoRA` with:
  - Rank `r = 4`
  - Alpha = 16
  - Dropout = 0.1
  - Target Modules: `query`, `value`

### Training Configuration
- Optimizer: AdamW
- Learning rate: 2e-4
- Weight decay: 0.01
- Scheduler: Cosine decay
- Epochs: 5
- Batch size: 16 (per device)
- Warmup steps: 500
- Mixed precision: Enabled (`fp16`)
- Logging every 100 steps

## 📈 Results

| Metric     | Value       |
|------------|-------------|
| Accuracy   | 94.53%      |
| Parameters | 741,124     |

Training was performed on the NYU HPC GPU cluster. LoRA provided high efficiency with fewer trainable parameters and minimal accuracy trade-offs.

## 🧪 Lessons Learned

- **LoRA Rank**: `r=4` gives a sweet spot between performance and efficiency.
- **Dropout**: 0.1 prevents overfitting; higher values degrade performance.
- **Warmup Steps**: 500 helped stabilize training; too high slowed convergence.
- **Batch Size**: 16 worked well for GPU utilization.

## 📦 Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/Spandan-2002/LoRA-BERT_News_classifier.git
cd LoRA-BERT_News_classifier
pip install -r requirements.txt
'''

## Contributing
Feel free to open issues or submit pull requests to improve the project.

