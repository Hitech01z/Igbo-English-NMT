import torch

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def get_token_id(tokenizer, names):

    for name in names:

        if hasattr(tokenizer, name):

            value = getattr(tokenizer, name)

            if callable(value):
                return value()

            return value

    if hasattr(tokenizer, "token_to_id"):

        mapping = {
            "sos_token_id": "<sos>",
            "bos_token_id": "<sos>",
            "start_token_id": "<sos>",
            "eos_token_id": "<eos>",
            "end_token_id": "<eos>",
        }

        for name in names:

            if name in mapping:

                token = mapping[name]

                if token in tokenizer.token_to_id:
                    return tokenizer.token_to_id[token]

    raise AttributeError(
        f"Tokenizer missing {names}"
    )


def create_padding_mask(sequence):
    return sequence == 0


def greedy_decode(
    model,
    src_tokens,
    target_tokenizer,
    max_length=100,
):

    model.eval()

    src_tokens = src_tokens.to(DEVICE)

    src_padding_mask = create_padding_mask(src_tokens)

    sos_id = get_token_id(
        target_tokenizer,
        [
            "sos_token_id",
            "bos_token_id",
            "start_token_id",
        ],
    )

    eos_id = get_token_id(
        target_tokenizer,
        [
            "eos_token_id",
            "end_token_id",
        ],
    )

    generated = torch.tensor(
        [[sos_id]],
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.no_grad():

        for _ in range(max_length):

            trg_padding_mask = create_padding_mask(
                generated
            )

            output = model(
                src=src_tokens,
                trg=generated,
                src_padding_mask=src_padding_mask,
                trg_padding_mask=trg_padding_mask,
            )

            next_token = (
                output[:, -1]
                .argmax(dim=-1)
                .item()
            )

            generated = torch.cat(
                (
                    generated,
                    torch.tensor(
                        [[next_token]],
                        device=DEVICE,
                    ),
                ),
                dim=1,
            )

            if next_token == eos_id:
                break

    return generated.squeeze(0).tolist()


def decode_tokens(
    token_ids,
    tokenizer,
):

    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids)

    words = []

    for token in token_ids:

        if token in [
            tokenizer.pad_token_id,
            tokenizer.sos_token_id,
            tokenizer.eos_token_id,
        ]:
            continue

        words.append(
            tokenizer.id_to_token[token]
        )

    return " ".join(words)