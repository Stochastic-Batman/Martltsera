import argparse
import logging
import os
import random
import time
import torch
import torch.nn as nn

from Gamarjoba import Gamarjoba
from get_data import get_dataset_pairs, ALL_GEORGIAN_CHARS


random.seed(95)  # ⚡
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger("MartltseraLogger (Train)")

# constants and hyperparameters
DICTIONARY_SIZE = 50000
HIDDEN_SIZE = 512
NUM_LAYERS = 1
DROPOUT_P = 0.2
LEARNING_RATE = 0.0001
TEACHER_FORCING_RATIO = 0.5
MODEL_SAVE_PATH = "../models/Martltsera_5.pth"
SOS_token = 0
EOS_token = 1

char_to_index = {char: i + 2 for i, char in enumerate(ALL_GEORGIAN_CHARS)}
index_to_char = {i + 2: char for i, char in enumerate(ALL_GEORGIAN_CHARS)}
VOCAB_SIZE = len(ALL_GEORGIAN_CHARS) + 2


# Design Considerations for Variable Output Length
# I adopted a classic encoder-decoder (seq2seq) architecture with no forced alignment.
# The encoder (LSTM) processes the entire misspelled input and compresses it into a fixed-size context (final hidden and cell states).
# The decoder (also LSTM) then autoregressively generates the corrected word one character at a time, starting from a <SOS> token and stopping when it predicts <EOS>.
# This design allows the output length to be completely independent of input length, naturally handling insertions, deletions, and substitutions without any padding or length constraints.
#
# Character Vocabulary and Handling the Georgian Alphabet
# The Georgian Mkhedruli script consists of 33 modern letters (U+10D0 to U+10F0, but contiguous from U+10D0='ა' to U+10FC='ჰ').
# In the provided code, ALL_GEORGIAN_CHARS = [chr(i) for i in range(4304, 4337)], which covers exactly these 33 characters (4304 is 10D0 in hex; 4336 is 10FC in hex).
# The vocabulary size is therefore 33 + 2 special tokens = 35.
# Character-to-index mapping assigns indices 2..34 to the 33 letters (in Unicode order), reserving 0 for <SOS> and 1 for <EOS>.
# During tokenization, any character not in this set is skipped (tensor_from_word filters with "if char in char_to_index"), which is safe because the dataset contains only Georgian words and the model focuses exclusively on Mkhedruli script.
#
# Special Tokens
# <SOS> (index 0): Used only to initialise the decoder at inference time (and implicitly during teacher forcing in training). It signals the start of generation and provides a neutral starting point for the decoder LSTM.
# <EOS> (index 1): Appended to every target sequence during training so the model learns to predict it after the last real character. At inference, greedy decoding stops when <EOS> is predicted, preventing over-generation.
# No <PAD> token: The model processes sequences one character at a time (batch_size=1 per word effectively, even in batches) and lengths vary naturally, so padding is unnecessary.


def tensor_from_word(word: str, device: torch.device) -> torch.Tensor:
    indexes = [char_to_index[char] for char in word if char in char_to_index]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def train_batch(model: Gamarjoba, optimizer: torch.optim.Optimizer, criterion: nn.NLLLoss, batch_pairs: list[tuple[str, str]], device: torch.device) -> float:
    optimizer.zero_grad()
    total_loss = torch.tensor(0.0, device=device)
    total_tokens = 0

    for input_word, target_word in batch_pairs:
        input_tensor = tensor_from_word(input_word, device)
        target_tensor = tensor_from_word(target_word, device)
        target_length = target_tensor.size(0)
        outputs = model(input_tensor, target_tensor, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
        example_loss = torch.tensor(0.0, device=device)

        for i in range(target_length):
            example_loss += criterion(outputs[i].unsqueeze(0), target_tensor[i])

        total_loss += example_loss
        total_tokens += target_length

    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        avg_loss.backward()
        optimizer.step()
        return avg_loss.item()

    optimizer.step()
    return 0.0


def validate(model: Gamarjoba, criterion: nn.NLLLoss, val_pairs: list[tuple[str, str]], device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for input_word, target_word in val_pairs:
            input_tensor = tensor_from_word(input_word, device)
            target_tensor = tensor_from_word(target_word, device)
            target_length = target_tensor.size(0)
            outputs = model(input_tensor, target_tensor, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
            example_loss = 0.0

            for i in range(target_length):
                example_loss += criterion(outputs[i].unsqueeze(0), target_tensor[i]).item()

            total_loss += example_loss
            total_tokens += target_length

    model.train()
    return total_loss / total_tokens if total_tokens > 0 else 0.0


def get_batches(pairs: list[tuple[str, str]], batch_size: int):
    for i in range(0, len(pairs), batch_size):
        yield pairs[i:i + batch_size]


def train_model(epochs: int, batch_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = Gamarjoba(VOCAB_SIZE, HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout_p=DROPOUT_P)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.NLLLoss()

    logger.info("Generating dataset...")
    pairs = get_dataset_pairs(dictionary_size=DICTIONARY_SIZE)
    correct_pairs = [(w, w) for _, w in random.sample(pairs, int(0.2 * len(pairs)))]  # add correct pairs
    pairs += correct_pairs
    random.shuffle(pairs)
    n = len(pairs)
    train_size = int(0.8 * n)
    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:]
    logger.info(f"Dataset ready: {len(train_pairs)} train pairs, {len(val_pairs)} val pairs.")

    start_time = time.time()
    log_interval = 200
    loss_interval = 0.0
    total_steps = epochs * len(train_pairs)
    iter_count = 0
    best_val_loss = float('inf')
    patience = 5
    counter = 0

    logger.info(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        random.shuffle(train_pairs)
        for batch_pairs in get_batches(train_pairs, batch_size):
            batch_len = len(batch_pairs)
            iter_count += batch_len
            loss = train_batch(model, optimizer, criterion, batch_pairs, device)
            loss_interval += loss * batch_len
            if iter_count % log_interval == 0:
                loss_avg = loss_interval / log_interval
                logger.info(f"{iter_count} steps ({iter_count / total_steps * 100:.0f}% complete) | Loss: {loss_avg:.3f}")
                loss_interval = 0.0

        val_loss = validate(model, criterion, val_pairs, device)
        logger.info(f"Epoch {epoch} Validation Loss: {val_loss:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH[:-4] + f"_{epoch}.pth"), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH[:-4] + f"_{epoch}.pth")
            logger.info(f"New best model saved to {MODEL_SAVE_PATH[:-4] + f"_{epoch}.pth"}")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    train_model(args.epochs, args.batch_size)