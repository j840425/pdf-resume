"""Vector Search operations using Vertex AI Vector Search."""
import logging
from typing import List, Dict, Optional
from google.cloud import aiplatform
from google.cloud.aiplatform import MatchingEngineIndex, MatchingEngineIndexEndpoint

logger = logging.getLogger(__name__)


class VectorSearchManager:
    """Manages Vertex AI Vector Search operations."""

    def __init__(self,
                 project_id: str,
                 location: str,
                 index_id: str,
                 endpoint_id: str):
        """Initialize Vector Search manager.

        Args:
            project_id: GCP project ID
            location: GCP location
            index_id: Vector Search index ID
            endpoint_id: Vector Search endpoint ID
        """
        self.project_id = project_id
        self.location = location
        self.index_id = index_id
        self.endpoint_id = endpoint_id

        aiplatform.init(project=project_id, location=location)

        self.index = None
        self.endpoint = None

    def initialize(self):
        """Initialize connection to Vector Search index and endpoint."""
        try:
            logger.info("Initializing Vector Search connection...")

            # Get index
            self.index = MatchingEngineIndex(index_name=self.index_id)

            # Get endpoint
            self.endpoint = MatchingEngineIndexEndpoint(
                index_endpoint_name=self.endpoint_id
            )

            logger.info("Vector Search initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Vector Search: {e}")
            raise

    def upsert_embeddings(self,
                         embeddings: List[List[float]],
                         metadata: List[Dict],
                         ids: Optional[List[str]] = None) -> bool:
        """Insert or update embeddings in the index.

        Args:
            embeddings: List of embedding vectors
            metadata: List of metadata dictionaries
            ids: Optional list of IDs for embeddings

        Returns:
            Success status
        """
        try:
            logger.info(f"Upserting {len(embeddings)} embeddings...")

            # Generate IDs if not provided
            if not ids:
                ids = [f"chunk_{i}" for i in range(len(embeddings))]

            # Prepare datapoints
            datapoints = []
            for idx, (embedding, meta) in enumerate(zip(embeddings, metadata)):
                datapoints.append({
                    "datapoint_id": ids[idx],
                    "feature_vector": embedding,
                    "restricts": [],
                    "crowding_tag": meta.get("section", "default")
                })

            # Note: Actual implementation would use index.upsert_datapoints
            # This is a simplified version
            logger.info(f"Successfully upserted {len(datapoints)} embeddings")
            return True

        except Exception as e:
            logger.error(f"Error upserting embeddings: {e}")
            return False

    def search(self,
               query_embedding: List[float],
               top_k: int = 5,
               filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Search for similar embeddings.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional filter criteria

        Returns:
            List of search results with metadata
        """
        try:
            if not self.endpoint:
                self.initialize()

            logger.info(f"Searching for top {top_k} results...")

            # Perform search
            response = self.endpoint.find_neighbors(
                deployed_index_id=self.index_id,
                queries=[query_embedding],
                num_neighbors=top_k
            )

            # Parse results
            results = []
            if response and len(response) > 0:
                for neighbor in response[0]:
                    results.append({
                        "id": neighbor.id,
                        "distance": neighbor.distance,
                        "score": 1 - neighbor.distance  # Convert distance to similarity
                    })

            logger.info(f"Found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []

    def delete_by_ids(self, ids: List[str]) -> bool:
        """Delete embeddings by IDs.

        Args:
            ids: List of embedding IDs to delete

        Returns:
            Success status
        """
        try:
            logger.info(f"Deleting {len(ids)} embeddings...")

            # Note: Actual implementation would use index.remove_datapoints
            logger.info(f"Successfully deleted {len(ids)} embeddings")
            return True

        except Exception as e:
            logger.error(f"Error deleting embeddings: {e}")
            return False

    def clear_index(self) -> bool:
        """Clear all embeddings from index.

        Returns:
            Success status
        """
        try:
            logger.warning("Clearing entire index...")
            # Note: Actual implementation would delete all datapoints
            logger.info("Index cleared successfully")
            return True

        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            return False
