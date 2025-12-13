import logging
import random
import torch
from Gamarjoba import Gamarjoba
from get_data import ALL_GEORGIAN_CHARS


random.seed(95)  # ⚡
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger("MartltseraLogger (Inference)")

# constants and hyperparameters
HIDDEN_SIZE = 512
NUM_LAYERS = 1
DROPOUT_P = 0.2
SOS_token = 0
EOS_token = 1

char_to_index = {char: i + 2 for i, char in enumerate(ALL_GEORGIAN_CHARS)}
index_to_char = {i + 2: char for i, char in enumerate(ALL_GEORGIAN_CHARS)}
VOCAB_SIZE = len(ALL_GEORGIAN_CHARS) + 2


class SpellChecker:
    def __init__(self, model_path: str = "../models/Martltsera.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Gamarjoba(VOCAB_SIZE, HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout_p=DROPOUT_P)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # important for dropout and inference behaviour


    def tensor_from_word(self, word: str) -> torch.Tensor:
        idxs = [char_to_index.get(char, EOS_token) for char in word if char in char_to_index]
        idxs.append(EOS_token)
        return torch.tensor(idxs, dtype=torch.long, device=self.device).view(-1, 1)


    @staticmethod  # PyCharm would not shut up
    def idx_to_char(idx: int) -> str:
        return index_to_char.get(idx, "")


    def fix(self, word: str) -> str:
        if not word.strip():  # edge case
            return word

        with torch.no_grad():
            input_tensor = self.tensor_from_word(word)

            encoder_hidden, encoder_cell = self.model.encoder.init_hidden(self.device)
            for i in range(input_tensor.size(0)):
                _, encoder_hidden, encoder_cell = self.model.encoder(input_tensor[i], encoder_hidden, encoder_cell)

            decoder_input = torch.tensor([[SOS_token]], device=self.device)
            decoder_hidden = encoder_hidden
            decoder_cell = encoder_cell

            decoded_chars = []
            while True:
                prediction, decoder_hidden, decoder_cell = self.model.decoder(decoder_input, decoder_hidden, decoder_cell)
                _, top_i = prediction.topk(1)
                idx = top_i.squeeze().item()

                if idx == EOS_token or len(decoded_chars) >= 50:  # safety limit
                    break

                decoded_chars.append(self.idx_to_char(idx))
                decoder_input = top_i.squeeze().detach()  # feed prediction back as next input (no teacher forcing at inference)

            return "".join(decoded_chars)


def correct_word(word: str, model_path: str) -> str:
    model = SpellChecker(model_path=model_path)
    return model.fix(word)


if __name__ == "__main__":
    checker = SpellChecker()
    test_words = [
        "გამრჯობა",      # common typo
        "გამარჯობა",      # correct
        "თბილისი",        # correct
        "პროგამა",        # missing რ
        "კომპიუტრი",      # common mistake
        "სიყვარულ",       # missing ი
        "მართლწერა",      # correct
    ]
    logger.info("Inference examples:")
    for w in test_words:
        logger.info(f"{w:15} -> {checker.fix(w)}")