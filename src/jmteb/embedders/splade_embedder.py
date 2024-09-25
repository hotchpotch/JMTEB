from __future__ import annotations

from os import PathLike
from pathlib import Path

import numpy as np
import torch
import tqdm
from loguru import logger
from scipy.sparse import load_npz, save_npz, vstack
from scipy.sparse.csr import csr_matrix
from yasem import SpladeEmbedder as YasemSpladeEmbedder

from jmteb.embedders.base import TextEmbedder


class SpladeEmbedder(TextEmbedder):
    """Splade embedder."""

    _chunk_size = 1024

    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int = 32,
        device: str | None = None,
        max_seq_length: int | None = 512,
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
        self.is_sparse_model = True
        self.convert_to_tensor = False
        self.convert_to_numpy = False

    def encode(
        self,
        text: str | list[str],
        prefix: str | None = None,
        show_progress_bar: bool = False,
    ) -> csr_matrix | torch.Tensor:
        is_single_text = isinstance(text, str)
        if is_single_text:
            text = [text]
        if prefix:
            text = [prefix + t for t in text]
        embeddings = self.model.encode(
            text,
            convert_to_numpy=False,
            batch_size=self.batch_size,
            device=self.device,  # type: ignore
            show_progress_bar=show_progress_bar,
            convert_to_csr_matrix=True,
        )
        if is_single_text:
            embeddings = embeddings[:1]  # csr_matrix の場合、単一行を取得

        if self.convert_to_tensor:
            embeddings = torch.Tensor(
                embeddings.toarray()  # type: ignore
            )  # csr_matrix を Tensor に変換

        return embeddings  # csr_matrix または torch.Tensor

    def get_output_dim(self) -> int:
        return self.model.vocab_size

    def _batch_encode_and_save_on_disk(
        self,
        text_list: list[str],
        save_path: str | PathLike[str],
        prefix: str | None = None,
        batch_size: int = 64,
        dtype: str = "float32",
    ) -> csr_matrix | torch.Tensor:
        """
        テキストのリストをエンコードし、memmap または csr_matrix を使用してディスクに保存します。

        Args:
            text_list (list[str]): テキストのリスト
            save_path (str): 埋め込みを保存するパス
            prefix (str, optional): エンコード時に使用するプレフィックス。デフォルトは None。
            dtype (str, optional): データ型。デフォルトは "float32"。
            batch_size (int): バッチサイズ。デフォルトは 64。
        """

        num_samples = len(text_list)
        output_dim = self.get_output_dim()

        if self.is_sparse_model:
            embeddings = []
        elif self.convert_to_numpy:
            embeddings = np.memmap(
                save_path, dtype=dtype, mode="w+", shape=(num_samples, output_dim)
            )
        else:
            embeddings = torch.empty(
                (num_samples, output_dim),
                dtype=self._torch_dtype_parser(dtype),  # type: ignore
            )

        with tqdm.tqdm(total=num_samples, desc=f"Encoding / bs {batch_size}") as pbar:
            for i in range(0, num_samples, batch_size):
                batch = text_list[i : i + batch_size]
                batch_embeddings: csr_matrix | torch.Tensor = self.encode(
                    batch, prefix=prefix
                )
                if self.is_sparse_model:
                    embeddings.append(batch_embeddings)  # type: ignore
                elif self.convert_to_numpy:
                    embeddings[i : i + batch_size] = batch_embeddings
                else:
                    embeddings[i : i + batch_size] = batch_embeddings
                pbar.update(len(batch))

        if self.is_sparse_model:
            # csr_matrix を縦方向に連結
            concatenated_embeddings = vstack(embeddings)
            save_npz(save_path, concatenated_embeddings)
            return concatenated_embeddings
        elif self.convert_to_numpy:
            embeddings.flush()  # type: ignore
            return np.memmap(
                save_path, dtype=dtype, mode="r", shape=(num_samples, output_dim)
            )
        else:
            torch.save(embeddings, save_path)
            return embeddings

    def batch_encode_with_cache(
        self,
        text_list: list[str],
        prefix: str | None = None,
        cache_path: str | PathLike[str] | None = None,
        overwrite_cache: bool = False,
        dtype: str = "float32",
    ) -> csr_matrix | torch.Tensor:
        """
        テキストのリストをエンコードし、cache_path が指定されている場合は memmap または csr_matrix を使用してディスクに保存します。

        Args:
            text_list (list[str]): テキストのリスト
            prefix (str, optional): エンコード時に使用するプレフィックス。デフォルトは None。
            cache_path (str, optional): 埋め込みを保存するパス。デフォルトは None。
            overwrite_cache (bool, optional): キャッシュを上書きするかどうか。デフォルトは False。
            dtype (str, optional): データ型。デフォルトは "float32"。
        """

        if (
            self.is_sparse_model
            and cache_path is not None
            and not str(cache_path).endswith(".npz")
        ):
            cache_path = str(cache_path) + ".npz"

        if cache_path is None:
            logger.info("Encoding embeddings")
            return self.encode(text_list, prefix=prefix)

        cache_path = Path(cache_path)

        if cache_path.exists() and not overwrite_cache:
            logger.info(f"Loading embeddings from {cache_path}")
            if self.is_sparse_model:
                # scipy の load_npz を使用して csr_matrix を読み込む
                return load_npz(cache_path)
            elif self.convert_to_numpy:
                return np.memmap(
                    cache_path,
                    dtype=dtype,
                    mode="r",
                    shape=(len(text_list), self.get_output_dim()),
                )
            else:
                return torch.load(cache_path)

        logger.info(f"Encoding and saving embeddings to {cache_path}")
        embeddings = self._batch_encode_and_save_on_disk(
            text_list,
            cache_path,
            prefix=prefix,
            batch_size=self._chunk_size,
            dtype=dtype,
        )

        return embeddings
