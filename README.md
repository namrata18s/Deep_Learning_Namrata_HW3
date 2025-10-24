# Deep_Learning_Namrata_HW3

# BERT Extractive Question Answering on Spoken-SQuAD

Deep Learning (CpSc 8430) - Homework 3  
Clemson University

## 📋 Overview

This project implements an extractive question answering system for the Spoken-SQuAD dataset using transformer-based models (BERT/RoBERTa). The system predicts answer spans within spoken documents that contain ASR transcription errors.

## 🎯 Task Description

**Input:** 
- Question (text)
- Context (ASR transcription with ~23% word error rate)

**Output:**
- Answer span (start and end positions in context)

**Challenge:** Handle noisy ASR transcriptions to extract correct answers

## 📊 Dataset

**Spoken-SQuAD**
- Training: 37,111 QA pairs
- Testing: 5,351 QA pairs  
- Word Error Rate: ~22-23%
- Source: [Spoken-SQuAD GitHub](https://github.com/chiahsuan156/Spoken-SQuAD)

Dataset files (not included in repo):
- `spoken_train-v1.1.json`
- `spoken_test-v1.1.json`

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch transformers tqdm
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA-capable GPU (recommended)

### Running the Code

**Option 1: Jupyter Notebook (Recommended for Palmetto)**

1. Open `bert_qa.ipynb` in JupyterHub
2. Upload data files to the same directory
3. Run all cells sequentially (Shift+Enter)
4. Results saved in `predictions.json`

**Option 2: Python Script**

```bash
# Ensure data files are in current directory
python bert_qa.py
```

### Expected Output

```
Loading model: roberta-base
✓ Loaded 37111 training samples
✓ Loaded 5351 test samples

Training on cuda
Epoch 1/3: 100%|██████████| Loss: 0.8234
Epoch 2/3: 100%|██████████| Loss: 0.5123
Epoch 3/3: 100%|██████████| Loss: 0.3456

Making predictions: 100%|██████████|
✓ Generated 5351 predictions

✅ All done! predictions.json created
```

## 🏗️ Model Architecture

### Approach
- **Base Model:** Pre-trained transformer (BERT/RoBERTa)
- **Task:** Token-level classification for answer span prediction
- **Fine-tuning:** Full model fine-tuning on Spoken-SQuAD

### Models Supported

| Model | Parameters | Speed | Performance |
|-------|------------|-------|-------------|
| bert-base-uncased | 110M | Fast | Good |
| roberta-base | 125M | Fast | Better |
| albert-base-v2 | 12M | Fastest | Good |
| bert-large-uncased | 340M | Slow | Best |

**Default:** `roberta-base` (best balance of speed and accuracy)

## ⚙️ Configuration

Edit the `CONFIG` dictionary in the code:

```python
CONFIG = {
    'model_name': 'roberta-base',     # Change model here
    'batch_size': 16,                  # Reduce if GPU memory error
    'num_epochs': 3,                   # More epochs = better but slower
    'learning_rate': 3e-5,             # Standard for BERT fine-tuning
    'max_len': 384,                    # Max sequence length
}
```

### Hyperparameters Used

- **Optimizer:** AdamW (weight decay = 0.01)
- **Learning Rate Schedule:** Linear warmup + decay
- **Warmup Steps:** 10% of total training
- **Gradient Clipping:** max_norm = 1.0
- **Loss Function:** Cross-entropy on start/end positions

## 📂 Project Structure

```
hDeep_Learning_Namrata_HW3/
├── bert_qa.ipynb           # Jupyter notebook version
├── predictions.json        # Model predictions (generated)
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

## 🎓 Implementation Details

### Dataset Processing
1. Load JSON data in SQuAD format
2. Tokenize questions and contexts
3. Convert answer character positions to token positions
4. Create PyTorch Dataset and DataLoader

### Training Loop
1. Forward pass through transformer
2. Compute cross-entropy loss on start/end logits
3. Backward pass with gradient clipping
4. Update weights with AdamW optimizer
5. Adjust learning rate with scheduler

### Prediction
1. Pass question-context pairs through model
2. Extract start and end logits
3. Take argmax for most likely positions
4. Decode tokens back to text
5. Post-process (ensure end ≥ start, limit length)

## 📈 Results

**Training Time:** ~30-40 minutes (V100 GPU)  
**Memory Usage:** ~8-10GB GPU memory  
**Test Predictions:** 5,351 answers generated  

### Baseline Achievement
- ✅ Simple Baseline (basic implementation)
- ✅ Medium Baseline (LR decay + model optimization)
- ✅ Strong Baseline (RoBERTa + hyperparameter tuning)

## 🔧 Troubleshooting

### Common Issues

**1. GPU Out of Memory**
```python
# Reduce batch size
'batch_size': 8,  # or even 4
```

**2. Model Not Found**
```bash
# Ensure internet connection for first download
# Models cached in ~/.cache/huggingface/
```

**3. Token Type IDs Error (RoBERTa)**
```
Already handled in code! RoBERTa doesn't use token_type_ids.
```

**4. CUDA Not Available**
```python
# Code automatically falls back to CPU
# Training will be much slower (~10x)
```

## 🚀 Advanced Usage

### Try Different Models

```python
# In CONFIG dictionary
'model_name': 'albert-base-v2',  # Efficient
'model_name': 'bert-large-uncased',  # Better performance
'model_name': 'distilbert-base-uncased',  # Faster
```

### Adjust Training

```python
# Longer training
'num_epochs': 5,

# Different learning rate
'learning_rate': 2e-5,

# Larger context window
'max_len': 512,
```

### Enable Mixed Precision (FP16)

For 2-3x speedup on V100/A100:
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(...)
    loss = outputs.loss

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 📊 Performance Optimization

### Speed Tips
1. Use V100 or A100 GPUs (much faster than P100)
2. Increase batch_size if GPU memory allows
3. Use DistilBERT for 2x speedup with minimal accuracy loss
4. Enable mixed precision training (FP16)

### Accuracy Tips
1. Use RoBERTa or ALBERT instead of BERT
2. Train for 4-5 epochs instead of 3
3. Try learning rate in range [2e-5, 5e-5]
4. Ensemble multiple models
5. Implement sliding window with doc_stride



## 👤 Author

Namrata Surve
nsurve@clemson.edu 
Clemson University - Fall 2025

## 🙏 Acknowledgments

- Hugging Face for Transformers library
- Spoken-SQuAD dataset creators
- Prof. Feng Luo for the course
- Clemson Palmetto Cluster for compute resources



---

**⭐ If you found this helpful, please star the repository!**
