"""Google Vertex embedding service (google-genai ``embed_content``).

Dograh's knowledge-base column is ``vector(1536)``, so ``gemini-embedding-001``
(3072 native dimensions) is asked for 1536 through ``output_dimensionality``.
Credentials and model metadata come from the Vertex catalogue; nothing here
logs or returns a credential.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from api.db.db_client import DBClient
from api.services.configuration.options.google_vertex_catalog import (
    get_vertex_model,
)

from .base import BaseEmbeddingService

DEFAULT_MODEL_ID = "gemini-embedding-001"
# Google allows 250 inputs / 20,000 tokens per request; chunks can be several
# hundred tokens each, so keep batches small enough to stay under the token cap.
MAX_BATCH_SIZE = 16


class VertexEmbeddingConfigError(Exception):
    """The Vertex embedding configuration cannot be used."""


class GoogleVertexEmbeddingService(BaseEmbeddingService):
    def __init__(
        self,
        db_client: DBClient,
        model_id: str = DEFAULT_MODEL_ID,
        project_id: Optional[str] = None,
        location: str = "global",
        credentials: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Any = None,
    ):
        entry = get_vertex_model("embeddings", model_id)
        if entry is None or not entry.dimensions:
            raise VertexEmbeddingConfigError(
                f"Vertex embedding model '{model_id}' is not supported by Dograh"
            )
        self.db = db_client
        self.model_id = model_id
        self._dimension = entry.dimensions
        self._project_id = project_id
        self._location = location or "global"
        self._credentials = credentials
        self._api_key = api_key
        self._client = client

    def _get_client(self):
        if self._client is None:
            # Imported lazily: the SDK is heavy and only needed on this path.
            from api.services.pipecat.service_factory import _vertex_client

            self._client = _vertex_client(
                api_key=self._api_key,
                credentials=self._credentials,
                project_id=self._project_id,
                location=self._location,
            )
        return self._client

    def get_model_id(self) -> str:
        return self.model_id

    def get_embedding_dimension(self) -> int:
        return self._dimension

    async def _embed(self, texts: List[str], task_type: str) -> List[List[float]]:
        from google.genai.types import EmbedContentConfig

        client = self._get_client()
        vectors: List[List[float]] = []
        for start in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[start : start + MAX_BATCH_SIZE]
            try:
                response = await client.aio.models.embed_content(
                    model=self.model_id,
                    contents=batch,
                    config=EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self._dimension,
                    ),
                )
            except Exception as e:
                # Class name only: SDK errors can embed request URLs.
                logger.error(
                    f"Vertex embeddings failed: model={self.model_id}, "
                    f"location={self._location}, error={type(e).__name__}"
                )
                raise
            batch_vectors = [list(item.values) for item in response.embeddings]
            if len(batch_vectors) != len(batch):
                raise ValueError(
                    f"Vertex returned {len(batch_vectors)} embeddings for "
                    f"{len(batch)} inputs"
                )
            for vector in batch_vectors:
                if len(vector) != self._dimension:
                    raise ValueError(
                        f"Vertex returned {len(vector)}-dimensional vectors, "
                        f"expected {self._dimension}"
                    )
            vectors.extend(batch_vectors)
        return vectors

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return await self._embed(texts, "RETRIEVAL_DOCUMENT")

    async def embed_query(self, query: str) -> List[float]:
        return (await self._embed([query], "RETRIEVAL_QUERY"))[0]

    async def search_similar_chunks(
        self,
        query: str,
        organization_id: int,
        limit: int = 5,
        document_uuids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        query_embedding = await self.embed_query(query)
        return await self.db.search_similar_chunks(
            query_embedding=query_embedding,
            organization_id=organization_id,
            limit=limit,
            document_uuids=document_uuids,
            embedding_model=self.model_id,
        )
