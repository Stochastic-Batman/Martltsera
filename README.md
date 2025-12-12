# Martltsera ✍️
Character-level Seq2Seq Georgian Spellchecker

Martltsera ("Spelling") is uses recurrent neural networks (GRU/LSTM) to correct misspelled Georgian words. It operates on a character level, learning the intrinsic orthography of the language to fix typos, missing characters, and keyboard slips.

## Project Structure

This project follows a **code-first** approach. The core logic is implemented in modular Python scripts within the `src/` directory, while Jupyter notebooks are used for demonstration and analysis.

```text
Martltsera/
├── src/
│   ├── Gamarjoba.py    # Defines the Seq2Seq GRU/LSTM Architecture
│   ├── get_data.py      # Handles synthetic error generation & data loading
│   ├── train.py        # Contains the training loop and saving logic
│   └── predict.py      # Inference logic for correcting words
├── notebooks/
│   ├── 1_training.ipynb   # Data generation & Model training
│   └── 2_inference.ipynb  # Interactive correction demo
└── models/                # Stores trained model weights
```

## Setup

Check your Python version with `python --version`. If it is not already Python 3.14, set it to 3.14. Then create a virtual environment with:

`python -m venv martltsera_venv`

and install requirements with:

`pip install -r requirements.txt`

## Usage

1. You can run the training script directly from the terminal:

```bash
# Trains the model and saves it to models/Martltsera.pth
python src/train.py --epochs 50 --batch_size 64
```

2. Correcting Words (Inference):

```Python
from src.predict import SpellChecker
corrector = SpellChecker(model_path='models/Martltsera.pth')
print(corrector.fix("გამრჯობა"))
```
