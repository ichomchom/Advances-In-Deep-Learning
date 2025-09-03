from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions step-by-step with reasoning, and always provides the final answer in a concise manner."},
            {"role": "user", "content": "Convert 2 hours to minutes"},
            {"role": "assistant", "content": "First, we know that 1 hour is equal to 60 minutes. Therefore, to convert 2 hours to minutes, we multiply 2 by 60. So, 2 hours is equal to 120 minutes. <answer>120</answer>"},
            {"role": "user", "content": "What is 5 gallons in ounces?"},
            {"role": "assistant", "content": "First, we know that 1 gallon is equal to 128 ounces. Therefore, to convert 5 gallons to ounces, we multiply 5 by 128. So, 5 gallons is equal to 640 ounces. <answer>640</answer>"},
            {"role": "user", "content": "How many inches is 6 feet?"},
            {"role": "assistant", "content": "First, we know that 1 foot is equal to 12 inches. Therefore, to convert 6 feet to inches, we multiply 6 by 12. So, 6 feet is equal to 72 inches. <answer>72</answer>"},
            {"role": "user", "content": question},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        return prompt


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
