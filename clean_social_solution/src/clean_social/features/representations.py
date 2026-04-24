from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_TOKEN_REGEX = re.compile(r"[a-zA-Z']+")


@dataclass
class TfidfBundle:
    matrix: np.ndarray
    feature_names: list[str]
    vectorizer: TfidfVectorizer


@dataclass
class GloveBundle:
    matrix: np.ndarray
    embedding_dim: int
    source: str


def build_tfidf_matrix(texts: Iterable[str], max_features: int = 3000) -> TfidfBundle:
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    sparse_matrix = vectorizer.fit_transform([str(text) for text in texts])
    matrix = sparse_matrix.toarray()
    return TfidfBundle(
        matrix=matrix,
        feature_names=vectorizer.get_feature_names_out().tolist(),
        vectorizer=vectorizer,
    )


def tokenize(text: str) -> list[str]:
    return _TOKEN_REGEX.findall(str(text).lower())


def _hash_embedding(token: str, dim: int) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(token)) % (2**32))
    return rng.normal(loc=0.0, scale=0.3, size=dim)


def _build_hash_embeddings(texts: list[str], dim: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    for text in texts:
        tokens = tokenize(text)
        if not tokens:
            rows.append(np.zeros(dim, dtype=np.float32))
            continue
        token_vectors = np.vstack([_hash_embedding(token, dim) for token in tokens])
        rows.append(token_vectors.mean(axis=0).astype(np.float32))
    return np.vstack(rows)


def _load_local_glove_subset(glove_path: Path, vocabulary: set[str], dim: int) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}

    with glove_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split(" ", 1)
            if len(parts) != 2:
                continue

            token, values = parts
            if token not in vocabulary:
                continue

            vector = np.fromstring(values, sep=" ", dtype=np.float32)
            if vector.shape[0] != dim:
                continue

            vectors[token] = vector

            if len(vectors) == len(vocabulary):
                break

    return vectors


def _build_glove_from_lookup(texts: list[str], dim: int, lookup: dict[str, np.ndarray]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for text in texts:
        tokens = tokenize(text)
        vectors = [lookup[token] for token in tokens if token in lookup]
        if vectors:
            rows.append(np.mean(vectors, axis=0).astype(np.float32))
        else:
            rows.append(np.zeros(dim, dtype=np.float32))
    return np.vstack(rows)


def _load_gensim_glove(dim: int):
    try:
        import gensim.downloader as api

        return api.load(f"glove-wiki-gigaword-{dim}")
    except Exception:
        return None


def build_glove_matrix(
    texts: Iterable[str],
    embedding_dim: int = 100,
    local_glove_path: str | Path | None = None,
) -> GloveBundle:
    text_list = [str(text) for text in texts]
    vocabulary = {token for text in text_list for token in tokenize(text)}

    if local_glove_path:
        glove_path = Path(local_glove_path)
        if glove_path.exists() and glove_path.is_file():
            local_lookup = _load_local_glove_subset(glove_path, vocabulary, embedding_dim)
            if local_lookup:
                return GloveBundle(
                    matrix=_build_glove_from_lookup(text_list, embedding_dim, local_lookup),
                    embedding_dim=embedding_dim,
                    source=f"local_glove_{glove_path.name}",
                )

    glove_model = _load_gensim_glove(embedding_dim)

    if glove_model is None:
        return GloveBundle(
            matrix=_build_hash_embeddings(text_list, embedding_dim),
            embedding_dim=embedding_dim,
            source="hash_fallback",
        )

    return GloveBundle(
        matrix=_build_glove_from_lookup(
            text_list,
            embedding_dim,
            {token: np.asarray(glove_model[token], dtype=np.float32) for token in vocabulary if token in glove_model},
        ),
        embedding_dim=embedding_dim,
        source=f"gensim_glove_{embedding_dim}",
    )
