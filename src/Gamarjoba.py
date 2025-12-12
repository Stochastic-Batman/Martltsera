import random
import torch
import torch.nn as nn


# LSTM (Long Short-Term Memory) Justification:
# I chose the LSTM architecture because Georgian words have a complex structure. A single word often consists of a root with many prefixes and suffixes attached to it.
# To fix a spelling error, the model needs to remember the beginning of the word (the root) to figure out the correct ending.
# Standard RNNs often struggle to remember this information over long sequences because of the "vanishing gradient" problem.
# LSTMs solve this using a Cell State - a type of internal memory that acts like a highway, allowing information to travel through the network without getting lost.
# This makes the LSTM perfect for learning the long and complex character patterns found in the Georgian language.
#
# Encoder-Decoder Architecture Justification:
# I implemented an Encoder-Decoder structure because spelling errors often change the length of a word.
# For example, if a user misses a key or types an extra one, the input length will differ from the correct output.
# This architecture solves that problem by separating the task into two parts.
# The Encoder reads the entire misspelled word first to capture its full context.
# The Decoder then uses that information to generate the correct word character by character.
# This allows the model to handle complex errors like insertions and deletions, not just simple letter replacements.

class EncoderLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout_p: float = 0.2):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(input_size, hidden_size)  # convert character indices to dense vectors
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)  # this saved me tons of time


    def forward(self, input_seq: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.embedding(input_seq).view(1, 1, -1)  # Shape (N) -> Shape (seq_len=1, batch_size=1, input_size=N); as 1 * 1  = 1, -1 will automatically infer the embedding dimension
        output = self.dropout(embedded)
        output, (hidden, cell) = self.lstm(output, (hidden, cell))  # updates hidden/cell states based on current input

        return output, hidden, cell

    # LSTM processes data sequentially. Since there is no previous memory at the start, we must create one: (h_0, c_0).
    def init_hidden(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device), torch.zeros(self.num_layers, 1, self.hidden_size, device=device)  # (num_layers,


class DecoderLSTM(nn.Module):
    def __init__(self, output_size: int, hidden_size: int, num_layers: int = 1, dropout_p: float = 0.2):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(output_size, hidden_size)  # embeds the previous character (or SOS)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)  # processes the sequence step-by-step
        self.out = nn.Linear(hidden_size, output_size)  # projects hidden state to vocabulary size (logits)
        self.softmax = nn.LogSoftmax(dim=1)  # converts logits to log-probabilities for prediction


    def forward(self, input_step: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.embedding(input_step).view(1, 1, -1)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)
        output, (hidden, cell) = self.lstm(output, (hidden, cell))  # updates state based on input and previous state
        prediction = self.softmax(self.out(output[0]))  # computes probability distribution over vocabulary

        return prediction, hidden, cell


class Gamarjoba(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int = 1, dropout_p: float = 0.2):
        super(Gamarjoba, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = EncoderLSTM(vocab_size, hidden_size, num_layers, dropout_p).to(self.device)
        self.decoder = DecoderLSTM(vocab_size, hidden_size, num_layers, dropout_p).to(self.device)
        self.vocab_size = vocab_size


    # Teacher Forcing: A training method where the model is sometimes fed the actual correct previous character instead of its own predicted guess to speed up convergence.
    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_hidden, encoder_cell = self.encoder.init_hidden(self.device)  # initializes hidden state with zeros

        for i in range(input_length):
            _, encoder_hidden, encoder_cell = self.encoder(input_tensor[i], encoder_hidden, encoder_cell)  # builds context vector

        decoder_input = torch.tensor([[0]], device=self.device)  # Start-Of-Sequence token assumed 0; starts decoding
        decoder_hidden = encoder_hidden  # passes the Encoder's final memory to Decoder
        decoder_cell = encoder_cell

        outputs = torch.zeros(target_length, self.vocab_size, device=self.device)

        for i in range(target_length):
            decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)  # predict next char
            outputs[i] = decoder_output  # stores the prediction

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False  # decides strategy randomly

            if use_teacher_forcing:
                decoder_input = target_tensor[i]  # feeds the correct character as next input
            else:
                _, top_i = decoder_output.topk(1)  # gets the character index with the highest probability
                decoder_input = top_i.squeeze().detach()  # detach() prevents backprop through this step (treats input as constant)
                if decoder_input.item() == 1:  # End-Of-Sequence token assumed 1
                    break

        return outputs