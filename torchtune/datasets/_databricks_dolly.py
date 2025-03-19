# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

from typing import Any, Callable, Dict, Optional, Union

from torchtune.data._messages import AlpacaToMessages

from torchtune.datasets._packed import PackedDataset
from torchtune.datasets._sft import SFTDataset
from torchtune.modules.transforms.tokenizers import ModelTokenizer


def databricks_dolly_15k(
    tokenizer: ModelTokenizer,
    *,
    source: str = "databricks/databricks-dolly-15k",
    max_seq_len: int = 512,
    packed: bool = False,
    column_map: Optional[Dict[str, str]] = {"input": "instruction", "output": "response"},
    train_on_input: bool = True,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],

    ) -> Union[SFTDataset, PackedDataset]:
    

    message_transform = AlpacaToMessages(
        train_on_input=train_on_input, column_map=column_map
    )
    ds = SFTDataset(
        source=source,
        message_transform=message_transform,
        model_transform=tokenizer,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs,
    )
    
    if packed:
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len)
    return ds

    """
    Support for family of Alpaca-style datasets from Hugging Face Datasets using
    the `data input format <https://huggingface.co/datasets/tatsu-lab/alpaca#data-instances>`_
    and `prompt template <https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py#L31>`_
    from the original alpaca codebase, where `instruction`, `input`, and `output`
    are fields from the dataset.

    Masking of the prompt during training is controlled by the `train_on_input` flag, which is
    set to `True` by `default <https://github.com/tloen/alpaca-lora/blob/main/finetune.py#L49>`_
    - If `train_on_input` is True, the prompt is used during training and
    contributes to the loss.
    - If `train_on_input` is False, the prompt is masked out (tokens replaced with -100)

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        source (str): path string of dataset, anything supported by Hugging Face's `load_dataset`.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is True.
        max_seq_len (int): Maximum number of tokens in the returned input and label token id lists.
            Default is 512, but we recommend setting this to the highest you can fit in memory and
            is supported by the model. For example, llama2-7B supports up to 4096 for sequence length.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.

    Returns:
        InstructDataset: dataset configured with source data and template


    Example:
        >>> alpaca_ds = alpaca_dataset(tokenizer=tokenizer)
        >>> for batch in Dataloader(alpaca_ds, batch_size=8):
        >>>     print(f"Batch size: {len(batch)}")
        >>> Batch size: 8
    """
"""
    return instruct_dataset(
        tokenizer=tokenizer,
        source=source,
        template="AlpacaInstructTemplate",
        train_on_input=train_on_input,
        max_seq_len=max_seq_len,
        packed=packed,
        split="train",
    )
"""

#alpaca_cleaned_dataset = partial(databricks_dolly_15k, source="yahma/alpaca-cleaned")
