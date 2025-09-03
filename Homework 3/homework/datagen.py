import json
import math
from .cot import CoTModel
from .data import Dataset, is_answer_valid
from tqdm import tqdm

def generate_dataset(output_json: str, oversample: int = 1, temperature: float = 0):
    model = CoTModel()
    dataset = Dataset("train")
    data = []
    success_count = 0
    total_count = 0

    for question, correct_answer in tqdm(dataset, desc="Generating RFT dataset"):
        total_count += 1

        # Generate multiple completions for the same question using proper formatting
        formatted_prompts = [model.format_prompt(question) for _ in range(oversample)]
        generations = model.batched_generate(
            formatted_prompts,
            num_return_sequences=oversample,
            temperature=temperature
        )

        for generation in generations:
            formatted_answer = model.parse_answer(generation)
            is_correct = is_answer_valid(formatted_answer, correct_answer)
            if is_correct:
                success_count += 1
                data.append([
                    question,
                   correct_answer,
                   generation
                ])

    success_rate = success_count / total_count * 100 if total_count > 0 else 0
    print(f"\nFinal success rate: {success_rate:.1f}% ({success_count}/{total_count})")

    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
