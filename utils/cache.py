"""Document caching utilities for persistence."""
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DocumentCache:
    """Manages caching of processed documents."""

    def __init__(self, cache_dir: str = ".cache"):
        """Initialize document cache.

        Args:
            cache_dir: Directory to store cached documents
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.metadata_file = self.cache_dir / "last_document_metadata.json"
        self.embeddings_file = self.cache_dir / "last_document_embeddings.pkl"

    def save_document(self, document_data: Dict) -> bool:
        """Save processed document to cache.

        Args:
            document_data: Complete document data including embeddings

        Returns:
            True if saved successfully
        """
        try:
            logger.info(f"Saving document to cache: {document_data.get('file_name')}")

            # Separate embeddings from metadata (embeddings are binary data)
            embeddings_data = {
                'chunks': []
            }

            metadata = {
                'file_name': document_data.get('file_name'),
                'metadata': document_data.get('metadata'),
                'structure': document_data.get('structure'),
                'hierarchical_structure': document_data.get('hierarchical_structure'),
                'summary_sections': document_data.get('summary_sections'),
                'summaries': document_data.get('summaries'),
                'overall_summary': document_data.get('overall_summary'),
                'cached_at': datetime.now().isoformat()
            }

            # Extract chunks with embeddings
            chunks_with_embeddings = document_data.get('chunks', [])
            chunks_without_embeddings = []

            for chunk in chunks_with_embeddings:
                # Store embedding separately
                if 'embedding' in chunk:
                    embeddings_data['chunks'].append(chunk['embedding'])

                # Store chunk metadata without embedding
                chunk_copy = {k: v for k, v in chunk.items() if k != 'embedding'}
                chunks_without_embeddings.append(chunk_copy)

            metadata['chunks'] = chunks_without_embeddings

            # Store pages_text if it exists
            if 'pages_text' in document_data:
                metadata['pages_text'] = document_data['pages_text']

            # Save metadata as JSON
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            # Save embeddings as pickle
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(embeddings_data, f)

            logger.info("Document saved to cache successfully")
            return True

        except Exception as e:
            logger.error(f"Error saving document to cache: {e}")
            return False

    def load_document(self) -> Optional[Dict]:
        """Load last processed document from cache.

        Returns:
            Document data or None if not found
        """
        try:
            # Check if cache files exist
            if not self.metadata_file.exists() or not self.embeddings_file.exists():
                logger.info("No cached document found")
                return None

            logger.info("Loading document from cache...")

            # Load metadata
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Load embeddings
            with open(self.embeddings_file, 'rb') as f:
                embeddings_data = pickle.load(f)

            # Merge embeddings back into chunks
            chunks = metadata.get('chunks', [])
            chunk_embeddings = embeddings_data.get('chunks', [])

            if len(chunks) == len(chunk_embeddings):
                for chunk, embedding in zip(chunks, chunk_embeddings):
                    chunk['embedding'] = embedding
            else:
                logger.warning(f"Mismatch in chunks ({len(chunks)}) and embeddings ({len(chunk_embeddings)})")

            metadata['chunks'] = chunks

            logger.info(f"Loaded cached document: {metadata.get('file_name')}")
            logger.info(f"Cached at: {metadata.get('cached_at')}")

            return metadata

        except Exception as e:
            logger.error(f"Error loading document from cache: {e}")
            return None

    def has_cached_document(self) -> bool:
        """Check if a cached document exists.

        Returns:
            True if cached document exists
        """
        return self.metadata_file.exists() and self.embeddings_file.exists()

    def get_cached_document_info(self) -> Optional[Dict]:
        """Get basic info about cached document without loading it.

        Returns:
            Basic document info or None
        """
        try:
            if not self.metadata_file.exists():
                return None

            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            return {
                'file_name': metadata.get('file_name'),
                'num_pages': metadata.get('metadata', {}).get('num_pages'),
                'num_sections': len(metadata.get('summaries', [])),
                'num_chunks': len(metadata.get('chunks', [])),
                'cached_at': metadata.get('cached_at')
            }

        except Exception as e:
            logger.error(f"Error getting cached document info: {e}")
            return None

    def clear_cache(self) -> bool:
        """Clear cached document.

        Returns:
            True if cleared successfully
        """
        try:
            if self.metadata_file.exists():
                self.metadata_file.unlink()
            if self.embeddings_file.exists():
                self.embeddings_file.unlink()

            logger.info("Cache cleared successfully")
            return True

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
