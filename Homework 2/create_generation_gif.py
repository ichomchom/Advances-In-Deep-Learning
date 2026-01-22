"""
Create a GIF showing the autoregressive image generation process.
This script generates images token-by-token and saves intermediate states as a GIF.
"""
from pathlib import Path
from typing import cast

import torch
from PIL import Image

from homework.autoregressive import Autoregressive
from homework.bsq import Tokenizer


def generate_with_frames(
    tokenizer_path: Path,
    autoregressive_path: Path,
    output_path: Path,
    n_images: int = 1,
    save_every: int = 10,
    duration: int = 50,
):
    """
    Generate images with intermediate frames for GIF creation.

    Args:
        tokenizer_path: Path to the tokenizer model
        autoregressive_path: Path to the autoregressive model
        output_path: Directory to save the GIF files
        n_images: Number of images to generate (default: 1)
        save_every: Save a frame every N tokens (default: 10)
        duration: Duration per frame in milliseconds (default: 50)
    """
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    print(f"Loading tokenizer from {tokenizer_path}")
    tk_model = cast(Tokenizer, torch.load(tokenizer_path, weights_only=False).to(device))

    print(f"Loading autoregressive model from {autoregressive_path}")
    ar_model = cast(Autoregressive, torch.load(autoregressive_path, weights_only=False).to(device))
    ar_model.eval()

    # Get dimensions
    dummy_index = tk_model.encode_index(torch.zeros(1, 100, 150, 3, device=device))
    _, h, w = dummy_index.shape
    seq_len = h * w

    print(f"Generating {n_images} image(s) with dimensions {h}x{w} ({seq_len} tokens)")
    print(f"Saving every {save_every} tokens")

    for img_idx in range(n_images):
        print(f"\nGenerating image {img_idx + 1}/{n_images}...")

        frames = []
        generated = torch.zeros(1, seq_len, dtype=torch.long, device=device)

        with torch.no_grad():
            # Generate token by token
            for t in range(seq_len):
                # Get embeddings
                x_emb = ar_model.embedding(generated)

                # Shift embeddings
                if t > 0:
                    x_emb_shifted = torch.nn.functional.pad(x_emb, (0, 0, 1, 0))[:, :-1, :]
                else:
                    x_emb_shifted = torch.zeros(1, 1, ar_model.d_latent, device=device)

                # Generate mask and transform
                mask = torch.nn.Transformer.generate_square_subsequent_mask(t + 1).to(device)
                x_transformed = ar_model.transformer(x_emb_shifted[:, :t+1], mask=mask)

                # Get next token
                logits = ar_model.output_projection(x_transformed[:, t])
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                generated[:, t] = next_token

                # Save frame at specified intervals or at the end
                if (t + 1) % save_every == 0 or t == seq_len - 1:
                    # Decode current state
                    current_tokens = generated.view(1, h, w)
                    current_image = tk_model.decode_index(current_tokens).cpu()

                    # Convert to PIL Image
                    np_image = (255 * (current_image[0] + 0.5).clip(0, 1)).to(torch.uint8).numpy()
                    pil_image = Image.fromarray(np_image)
                    frames.append(pil_image)

                    if (t + 1) % 100 == 0 or t == seq_len - 1:
                        print(f"  Token {t + 1}/{seq_len} - Frame {len(frames)}")

        # Save as GIF
        gif_path = output_path / f"generation_{img_idx}.gif"
        print(f"Saving GIF with {len(frames)} frames to {gif_path}")

        # Save the GIF with loop
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,  # 0 means infinite loop
        )

        # Also save the final frame as PNG
        final_png = output_path / f"generation_{img_idx}_final.png"
        frames[-1].save(final_png)
        print(f"Saved final frame to {final_png}")

    print(f"\nDone! Generated {n_images} GIF(s) in {output_path}")


if __name__ == "__main__":
    from fire import Fire
    Fire(generate_with_frames)
