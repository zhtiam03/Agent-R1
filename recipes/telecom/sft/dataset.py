"""Completion-only supervision using the exact inference prompt prefix."""

from recipes.telecom.tools import tool_schemas


def encode_step(tokenizer, row, max_length):
    prompt_ids = list(tokenizer.apply_chat_template(
        row["messages"], tools=tool_schemas(), add_generation_prompt=True, tokenize=True,
    ))
    # Encode the complete assistant message with its EOS/control tokens. Never
    # silently train a different template prefix or truncate away the completion.
    full_ids = list(tokenizer.apply_chat_template(
        row["messages"] + [{"role": "assistant", "content": row["action"]}],
        tools=tool_schemas(), add_generation_prompt=False, tokenize=True,
    ))
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat template does not preserve the generation prefix; add a tested model-specific formatter")
    if len(full_ids) > max_length or len(full_ids) <= len(prompt_ids):
        raise ValueError("SFT sample has no completion or exceeds max_length")
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids),
            "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]}


def collate_steps(examples, pad_token_id):
    """Pure-list collator; keeps EOS supervised even when pad_token == EOS."""
    length = max(len(row["input_ids"]) for row in examples)
    return {key: [row[key] + [padding] * (length - len(row[key])) for row in examples]
            for key, padding in (("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100))}
