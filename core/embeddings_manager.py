"""Embeddings generation and management."""
import logging
from typing import List, Dict
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingsManager:
    """Manages text embeddings using Vertex AI."""

    def __init__(self, vertex_service, model_name: str = "text-embedding-004"):
        """Initialize embeddings manager.

        Args:
            vertex_service: VertexService instance
            model_name: Name of the embedding model
        """
        self.vertex_service = vertex_service
        self.model_name = model_name

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for list of texts.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        logger.info(f"Generating embeddings for {len(texts)} texts...")

        try:
            embeddings = self.vertex_service.generate_embeddings(
                texts,
                model_name=self.model_name
            )

            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def generate_batch_embeddings(self,
                                  texts: List[str],
                                  batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings in batches.

        Args:
            texts: List of text strings
            batch_size: Number of texts per batch

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1} ({len(batch)} texts)")

            try:
                embeddings = self.generate_embeddings(batch)
                all_embeddings.extend(embeddings)
            except Exception as e:
                logger.error(f"Error in batch {i // batch_size + 1}: {e}")
                # Add placeholder embeddings for failed batch
                all_embeddings.extend([[0.0] * 768] * len(batch))

        return all_embeddings

    def compute_similarity(self,
                          embedding1: List[float],
                          embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score (0-1)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        return float(similarity)

    def find_most_similar(self,
                         query_embedding: List[float],
                         candidate_embeddings: List[List[float]],
                         top_k: int = 5) -> List[tuple]:
        """Find most similar embeddings to query.

        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return

        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = []

        for idx, candidate in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate)
            similarities.append((idx, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]
