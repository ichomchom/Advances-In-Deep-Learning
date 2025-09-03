from .base_llm import BaseLLM
from .sft import test_model, Dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import TrainingArguments, Trainer

def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm

def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length",
                     truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str, reason: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    import re
    
    # Extract answer from reason if it contains <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', reason)
    if answer_match:
        extracted_answer = answer_match.group(1)
        rounded_answer = round(float(extracted_answer), 3)
    else:
        rounded_answer = round(float(answer), 3)
    
    return {
        "question": prompt,
        "answer": f"<reason>{reason}</reason><answer>{rounded_answer}</answer>",
        "reason": f"<reason>{reason}</reason>"
    }


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, question=formated_data["question"], answer=formated_data["answer"])


def train_model(
    output_dir: str,
    **kwargs,
):
    # Reuse much of the SFT code here
    config = LoraConfig(
        r=16,
        lora_alpha=80,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    llm = BaseLLM()
    llm.model = get_peft_model(llm.model, config)

    llm.model.enable_input_require_grads()

    trainset = Dataset("rft")
    tokenized_train = TokenizedDataset(llm.tokenizer, trainset, format_example)

    args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        per_device_train_batch_size=32,
        num_train_epochs=10,
        learning_rate=1e-3,
        gradient_checkpointing=True,
        fp16=True,
        save_strategy="epoch",
        logging_steps=50,
    )

    trainer = Trainer(
        model=llm.model,
        args=args,
        train_dataset=tokenized_train,
    )

    trainer.train()

    trainer.save_model(output_dir)
    llm.model.save_pretrained(output_dir)
    llm.tokenizer.save_pretrained(output_dir)

    test_model(output_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
