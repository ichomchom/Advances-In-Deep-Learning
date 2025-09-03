import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.d_latent = d_latent
        self.n_tokens = n_tokens

        self.embedding = torch.nn.Embedding(n_tokens, d_latent)
        self.output_projection = torch.nn.Linear(d_latent, n_tokens)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_latent, nhead=8, dim_feedforward=512, batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=6
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, h, w = x.shape
        seq_len = h * w

        x_flat = x.view(B, seq_len)

        x_embedded = self.embedding(x_flat)

        x_shifted = torch.nn.functional.pad(
            x_embedded, (0, 0, 1, 0), value=0.0
        )[:, :-1, :]

        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            seq_len).to(x.device)

        x_transformed = self.transformer(x_shifted, mask=causal_mask)

        x_output = self.output_projection(x_transformed)

        output = x_output.view(B, h, w, self.n_tokens)
        return output, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        self.eval()
        seq_len = h * w
        generated = torch.zeros(B, seq_len, dtype=torch.long, device=device)

        with torch.no_grad():
            for t in range(seq_len):
                x_emb = self.embedding(generated)  # (B, t+1, d_latent)

                if t > 0:
                    x_emb_shifted = torch.nn.functional.pad(
                        x_emb, (0, 0, 1, 0))[:, :-1, :]
                else:
                    x_emb_shifted = torch.zeros(
                        B, 1, self.d_latent, device=device)

                mask = torch.nn.Transformer.generate_square_subsequent_mask(
                    t + 1).to(device)
                x_transformed = self.transformer(
                    x_emb_shifted[:, :t+1], mask=mask)  # (B, t+1, d_latent)

                logits = self.output_projection(
                    x_transformed[:, t])  # (B, n_tokens)
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(
                    probs, num_samples=1).squeeze(-1)  # (B,)
                generated[:, t] = next_token

        return generated.view(B, h, w)
