from __future__ import annotations

from os import PathLike

import numpy as np
import torch
from yasem import SpladeEmbedder as YasemSpladeEmbedder

from jmteb.embedders.base import TextEmbedder


class SpladeEmbedder(TextEmbedder):
    """Splade embedder."""

    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int = 32,
        device: str | None = None,
        max_seq_length: int | None = 512,
        add_eos: bool = False,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
    ) -> None:
        model_kwargs = self._model_kwargs_parser(model_kwargs)
        self.model = YasemSpladeEmbedder(
            model_name_or_path,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            max_seq_length=max_seq_length,
            device=device,  # type: ignore
        )
        if max_seq_length:
            self.model.max_seq_length = max_seq_length

        self.batch_size = batch_size
        self.device = device
        self.max_seq_length = self.model.max_seq_length
        self.add_eos = add_eos

    def encode(
        self,
        text: str | list[str],
        prefix: str | None = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray | torch.Tensor:
        if self.add_eos:
            text = self._add_eos_func(text)
        is_single_text = isinstance(text, str)
        if is_single_text:
            text = [text]
        if prefix:
            text = [prefix + t for t in text]
        embeddings = self.model.encode(
            text,
            convert_to_numpy=True,
            batch_size=self.batch_size,
            device=self.device,  # type: ignore
            show_progress_bar=show_progress_bar,
        )
        # print(self.model.get_token_values(embeddings))
        if is_single_text:
            embeddings = embeddings[0]
        if self.set_output_tensor:
            embeddings = torch.Tensor(embeddings)
        return embeddings  # type: ignore

    def _add_eos_func(self, text: str | list[str]) -> str | list[str]:
        try:
            eos_token = getattr(self.model.tokenizer, "eos_token")
        except AttributeError:
            return text

        if isinstance(text, str):
            return text + eos_token
        elif isinstance(text, list):
            return [t + eos_token for t in text]

    def get_output_dim(self) -> int:
        return self.model.vocab_size
