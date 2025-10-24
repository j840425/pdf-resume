"""Vertex AI service wrapper for all AI operations."""
import logging
from typing import List, Optional
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel
import vertexai

logger = logging.getLogger(__name__)


class VertexService:
    """Wrapper for Vertex AI services."""

    def __init__(self,
                 project_id: str,
                 location: str,
                 model_name: str = "gemini-1.5-pro",
                 embedding_model_name: str = "text-embedding-004"):
        """Initialize Vertex AI service.

        Args:
            project_id: GCP project ID
            location: GCP location
            model_name: Generative model name
            embedding_model_name: Embedding model name
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name

        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)

        self.generative_model = None
        self.embedding_model = None

    def initialize(self):
        """Initialize AI models."""
        try:
            logger.info("Initializing Vertex AI models...")

            # Initialize generative model
            self.generative_model = GenerativeModel(self.model_name)

            # Initialize embedding model
            self.embedding_model = TextEmbeddingModel.from_pretrained(
                self.embedding_model_name
            )

            logger.info("Vertex AI models initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing Vertex AI: {e}")
            raise

    def generate_text(self,
                     prompt: str,
                     temperature: float = 0.7,
                     max_output_tokens: int = 2048) -> str:
        """Generate text using Gemini.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_output_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        if not self.generative_model:
            self.initialize()

        try:
            logger.info(f"Generating text (temp={temperature}, max_tokens={max_output_tokens})")

            response = self.generative_model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                    "top_p": 0.95,
                    "top_k": 40,
                }
            )

            # Log response metadata
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    logger.info(f"Finish reason: {candidate.finish_reason}")
                if hasattr(candidate, 'safety_ratings'):
                    logger.debug(f"Safety ratings: {candidate.safety_ratings}")

            response_text = response.text
            logger.info(f"Generated {len(response_text)} characters")

            return response_text

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise

    def generate_embeddings(self,
                           texts: List[str],
                           model_name: Optional[str] = None) -> List[List[float]]:
        """Generate embeddings for texts.

        Args:
            texts: List of text strings
            model_name: Optional override for embedding model

        Returns:
            List of embedding vectors
        """
        if not self.embedding_model:
            self.initialize()

        try:
            logger.debug(f"Generating embeddings for {len(texts)} texts")

            # Generate embeddings
            embeddings = self.embedding_model.get_embeddings(texts)

            # Extract vectors
            vectors = [emb.values for emb in embeddings]

            return vectors

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def generate_text_streaming(self,
                               prompt: str,
                               temperature: float = 0.7,
                               max_output_tokens: int = 2048):
        """Generate text with streaming response.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_output_tokens: Maximum tokens to generate

        Yields:
            Text chunks as they are generated
        """
        if not self.generative_model:
            self.initialize()

        try:
            logger.debug("Starting streaming generation")

            response = self.generative_model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                },
                stream=True
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Input text

        Returns:
            Token count
        """
        if not self.generative_model:
            self.initialize()

        try:
            count_result = self.generative_model.count_tokens(text)
            return count_result.total_tokens

        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            # Fallback estimation
            return len(text) // 4
