"""Exercise the actual vec0 query with current and stale source generations."""

import json
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from basic_memory.repository.semantic_vector_index import VectorIndexScope, VectorKey
from basic_memory.repository.sqlite_vec_index import SQLiteVecIndex


@pytest.mark.asyncio
async def test_knn_auxiliary_hash_filter_keeps_only_current_project_vectors(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'vectors.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    index = SQLiteVecIndex(
        sessions,
        VectorIndexScope(namespace="test", project_id=1, embedding_identity="test:2", dimensions=2),
    )
    try:
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "CREATE TABLE search_vector_chunks ("
                    "id INTEGER PRIMARY KEY, entity_id INTEGER, project_id INTEGER, "
                    "chunk_key TEXT, source_hash TEXT, vector_index TEXT, "
                    "embedding_status TEXT, embedding_model TEXT)"
                )
            )
        await index.initialize()
        async with sessions() as session:
            # All vectors are equally near. Manifest filters, including generation
            # equality, must run without vec0 rejecting an auxiliary-column constraint.
            for row_id, project_id, manifest_hash, vector_hash in [
                (1, 1, "current", "current"),
                (2, 1, "new-generation", "old-generation"),
                (3, 2, "current", "current"),
            ]:
                await session.execute(
                    text(
                        "INSERT INTO search_vector_chunks VALUES "
                        "(:id, :id, :project, :key, :hash, 'sqlite-vec', 'ready', 'test:2')"
                    ),
                    {
                        "id": row_id,
                        "project": project_id,
                        "key": f"entity:{row_id}:0",
                        "hash": manifest_hash,
                    },
                )
                await session.execute(
                    text(
                        "INSERT INTO search_vector_embeddings(rowid, embedding, source_hash) "
                        "VALUES (:id, :vector, :hash)"
                    ),
                    {"id": row_id, "vector": json.dumps([1.0, 0.0]), "hash": vector_hash},
                )
            await session.commit()

        matches = await index.search([1.0, 0.0], limit=10)
        assert [match.key for match in matches] == [VectorKey(entity_id=1, chunk_key="entity:1:0")]
    finally:
        await engine.dispose()
