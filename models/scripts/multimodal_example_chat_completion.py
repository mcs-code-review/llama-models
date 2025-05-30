# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from io import BytesIO
from pathlib import Path
from typing import Optional

import fire
from llama_models.llama3.api.datatypes import RawMediaItem, RawMessage, RawTextItem

from llama_models.llama3.reference_impl.generation import Llama

THIS_DIR = Path(__file__).parent


def run_main(
    ckpt_dir: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    model_parallel_size: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=model_parallel_size,
    )

    # image understanding
    dialogs = []
    with open(THIS_DIR / "resources/dog.jpg", "rb") as f:
        img = f.read()

    dialogs = [
        [
            RawMessage(
                role="user",
                content=[
                    RawMediaItem(data=BytesIO(img)),
                    RawTextItem(text="Describe this image in two sentences"),
                ],
            )
        ],
    ]
    # text only
    dialogs += [
        [
            RawMessage(
                role="user",
                content="what is the recipe of mayonnaise in two sentences?",
            )
        ],
    ]

    for dialog in dialogs:
        result = generator.chat_completion(
            dialog,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for msg in dialog:
            print(f"{msg.role.capitalize()}: {msg.content}\n")

        out_message = result.generation
        print(f"> {out_message.role.capitalize()}: {out_message.content}")
        for t in out_message.tool_calls:
            print(f"  Tool call: {t.tool_name} ({t.arguments})")
        print("\n==================================\n")


def main():
    fire.Fire(run_main)


if __name__ == "__main__":
    main()
