"""Question-Answering engine for document queries."""
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class QAEngine:
    """Handles Q&A functionality for documents."""

    def __init__(self,
                 vertex_service,
                 embeddings_manager,
                 vector_search_manager):
        """Initialize Q&A engine.

        Args:
            vertex_service: VertexService instance
            embeddings_manager: EmbeddingsManager instance
            vector_search_manager: VectorSearchManager instance
        """
        self.vertex_service = vertex_service
        self.embeddings_manager = embeddings_manager
        self.vector_search_manager = vector_search_manager

    def answer_question(self,
                       question: str,
                       document_chunks: List[Dict],
                       top_k: int = 5,
                       summaries: Optional[List[Dict]] = None,
                       structure: Optional[Dict] = None) -> Dict:
        """Answer a question based on document content.

        Args:
            question: User's question
            document_chunks: List of document chunks with metadata
            top_k: Number of relevant chunks to retrieve
            summaries: Optional list of section summaries for additional context
            structure: Optional document structure with sections

        Returns:
            Dictionary with answer and source references
        """
        logger.info(f"Answering question: {question}")

        try:
            # Check if question refers to specific section by number/position
            target_section = self._identify_target_section(question, structure, summaries)

            # Generate embedding for question
            question_embedding = self.embeddings_manager.generate_embeddings([question])[0]

            # If specific section identified, prioritize its content
            if target_section:
                logger.info(f"Question targets specific section: {target_section['title']}")
                relevant_chunks = self._retrieve_chunks_from_section(
                    target_section,
                    document_chunks,
                    top_k
                )
                # Use the target section summary
                relevant_summaries = [s for s in (summaries or [])
                                     if s['section']['title'] == target_section['title']]
            else:
                # Search for relevant chunks normally
                relevant_chunks = self._retrieve_relevant_chunks(
                    question_embedding,
                    document_chunks,
                    top_k
                )

                # Find relevant summaries if provided
                relevant_summaries = []
                if summaries:
                    relevant_summaries = self._find_relevant_summaries(
                        question,
                        summaries,
                        max_summaries=3
                    )

            # Generate answer using retrieved context + summaries
            answer = self._generate_answer(question, relevant_chunks, relevant_summaries)

            # Extract source references
            sources = self._extract_sources(relevant_chunks)

            return {
                "answer": answer,
                "sources": sources,
                "relevant_chunks": relevant_chunks,
                "relevant_summaries": relevant_summaries
            }

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "answer": f"Error al procesar la pregunta: {str(e)}",
                "sources": [],
                "relevant_chunks": []
            }

    def _identify_target_section(self,
                                 question: str,
                                 structure: Optional[Dict],
                                 summaries: Optional[List[Dict]]) -> Optional[Dict]:
        """Identify if question refers to a specific section.

        Args:
            question: User's question
            structure: Document structure
            summaries: List of summaries

        Returns:
            Target section dict or None
        """
        if not summaries:
            return None

        import re

        question_lower = question.lower()

        # Patterns to detect section references
        patterns = [
            r'(?:la\s+)?primera\s+(?:sección|seccion)',  # "la primera sección"
            r'(?:la\s+)?segunda\s+(?:sección|seccion)',  # "la segunda sección"
            r'(?:la\s+)?tercera\s+(?:sección|seccion)',  # "la tercera sección"
            r'(?:la\s+)?cuarta\s+(?:sección|seccion)',   # "la cuarta sección"
            r'(?:la\s+)?quinta\s+(?:sección|seccion)',   # "la quinta sección"
            r'(?:la\s+)?(?:sección|seccion)\s+(\d+)',    # "sección 3"
            r'(?:la\s+)?(?:sección|seccion)\s+número\s+(\d+)',  # "sección número 3"
        ]

        # Ordinal words mapping
        ordinals = {
            'primera': 1, 'primer': 1, 'primero': 1,
            'segunda': 2, 'segundo': 2,
            'tercera': 3, 'tercer': 3, 'tercero': 3,
            'cuarta': 4, 'cuarto': 4,
            'quinta': 5, 'quinto': 5,
            'sexta': 6, 'sexto': 6,
            'séptima': 7, 'septima': 7, 'séptimo': 7, 'septimo': 7,
            'octava': 8, 'octavo': 8,
            'novena': 9, 'noveno': 9,
            'décima': 10, 'decima': 10, 'décimo': 10, 'decimo': 10,
        }

        section_index = None

        # Check ordinal patterns
        for ordinal_word, index in ordinals.items():
            if ordinal_word in question_lower:
                section_index = index
                break

        # Check numeric patterns
        if not section_index:
            for pattern in patterns:
                match = re.search(pattern, question_lower)
                if match:
                    groups = match.groups()
                    if groups and groups[0]:
                        section_index = int(groups[0])
                        break

        # If section index found, return corresponding section
        if section_index and 1 <= section_index <= len(summaries):
            target_summary = summaries[section_index - 1]
            return target_summary['section']

        return None

    def _retrieve_chunks_from_section(self,
                                      section: Dict,
                                      document_chunks: List[Dict],
                                      top_k: int) -> List[Dict]:
        """Retrieve chunks from a specific section.

        Args:
            section: Target section metadata
            document_chunks: All document chunks
            top_k: Number of chunks to retrieve

        Returns:
            List of chunks from the section
        """
        # Filter chunks by section page range
        start_page = section['start_page']
        end_page = section['end_page']

        section_chunks = [
            chunk for chunk in document_chunks
            if start_page <= chunk.get('page', 0) <= end_page
        ]

        # Return up to top_k chunks
        return section_chunks[:top_k] if section_chunks else document_chunks[:top_k]

    def _retrieve_relevant_chunks(self,
                                  query_embedding: List[float],
                                  document_chunks: List[Dict],
                                  top_k: int) -> List[Dict]:
        """Retrieve most relevant chunks for query.

        Args:
            query_embedding: Query embedding vector
            document_chunks: Available document chunks
            top_k: Number of chunks to retrieve

        Returns:
            List of relevant chunks with scores
        """
        # Extract embeddings from chunks
        chunk_embeddings = [chunk.get('embedding', []) for chunk in document_chunks]

        # Find most similar
        similar_indices = self.embeddings_manager.find_most_similar(
            query_embedding,
            chunk_embeddings,
            top_k
        )

        # Get corresponding chunks
        relevant_chunks = []
        for idx, score in similar_indices:
            chunk = document_chunks[idx].copy()
            chunk['relevance_score'] = score
            relevant_chunks.append(chunk)

        return relevant_chunks

    def _find_relevant_summaries(self,
                                 question: str,
                                 summaries: List[Dict],
                                 max_summaries: int = 3) -> List[Dict]:
        """Find summaries most relevant to the question.

        Args:
            question: User's question
            summaries: List of section summaries
            max_summaries: Maximum number of summaries to return

        Returns:
            List of relevant summaries
        """
        # Simple keyword matching for now - could be improved with embeddings
        question_lower = question.lower()

        scored_summaries = []
        for summary_item in summaries:
            section = summary_item['section']
            summary_text = summary_item['summary']

            # Score based on keyword overlap
            title_lower = section['title'].lower()
            summary_lower = summary_text.lower()

            score = 0
            # Check if question keywords appear in title or summary
            question_words = set(question_lower.split())
            title_words = set(title_lower.split())
            summary_words = set(summary_lower.split())

            # Higher weight for title matches
            title_overlap = len(question_words & title_words)
            summary_overlap = len(question_words & summary_words)

            score = title_overlap * 2 + summary_overlap * 0.5

            if score > 0:
                scored_summaries.append({
                    'summary_item': summary_item,
                    'score': score
                })

        # Sort by score and return top N
        scored_summaries.sort(key=lambda x: x['score'], reverse=True)
        return [item['summary_item'] for item in scored_summaries[:max_summaries]]

    def _generate_answer(self,
                        question: str,
                        relevant_chunks: List[Dict],
                        relevant_summaries: List[Dict] = None) -> str:
        """Generate answer using LLM and relevant context.

        Args:
            question: User's question
            relevant_chunks: Retrieved relevant chunks
            relevant_summaries: Retrieved relevant section summaries

        Returns:
            Generated answer
        """
        # Build context from chunks
        context = self._build_context(relevant_chunks)

        # Build summaries context if available
        summaries_context = ""
        if relevant_summaries:
            summaries_context = "\n\nRESÚMENES DE SECCIONES RELEVANTES:\n"
            for summary_item in relevant_summaries:
                section = summary_item['section']
                summary = summary_item['summary']
                summaries_context += f"\n**{section['title']}** (Páginas {section['start_page']}-{section['end_page']}):\n{summary}\n"

        prompt = f"""Responde la siguiente pregunta basándote en el contexto proporcionado.

CONTEXTO DETALLADO (Texto del documento):
{context}
{summaries_context}

PREGUNTA: {question}

INSTRUCCIONES:
- Responde de manera precisa y completa
- Usa información tanto del texto detallado como de los resúmenes de secciones
- Los resúmenes te dan una visión general; el texto detallado te da información específica
- Si el contexto no contiene información suficiente, indícalo claramente
- Menciona las secciones/páginas relevantes cuando sea apropiado
- Proporciona una respuesta bien estructurada y coherente
- Responde en español

RESPUESTA:"""

        try:
            answer = self.vertex_service.generate_text(
                prompt,
                max_output_tokens=65535  # Maximum for Gemini 2.5 Pro (includes thinking tokens)
            )
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error al generar respuesta: {str(e)}"

    def _build_context(self, chunks: List[Dict], max_length: int = 4000) -> str:
        """Build context string from chunks.

        Args:
            chunks: List of relevant chunks
            max_length: Maximum context length

        Returns:
            Context string
        """
        context_parts = []
        current_length = 0

        for chunk in chunks:
            section = chunk.get('section', 'Desconocida')
            page = chunk.get('page', '?')
            content = chunk.get('content', '')

            chunk_text = f"[Sección: {section}, Página: {page}]\n{content}\n"

            if current_length + len(chunk_text) > max_length:
                break

            context_parts.append(chunk_text)
            current_length += len(chunk_text)

        return "\n---\n".join(context_parts)

    def _extract_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Extract source references from chunks.

        Args:
            chunks: Relevant chunks

        Returns:
            List of source references
        """
        sources = []

        for chunk in chunks:
            sources.append({
                "section": chunk.get('section', 'Desconocida'),
                "page": chunk.get('page', '?'),
                "relevance": chunk.get('relevance_score', 0.0)
            })

        return sources

    def generate_suggested_questions(self,
                                    document_summary: str,
                                    num_questions: int = 5) -> List[str]:
        """Generate suggested questions based on document.

        Args:
            document_summary: Summary of the document
            num_questions: Number of questions to generate

        Returns:
            List of suggested questions
        """
        prompt = f"""Basándote en el siguiente resumen de documento, genera {num_questions} preguntas relevantes que un usuario podría querer hacer sobre el contenido.

RESUMEN DEL DOCUMENTO:
{document_summary[:2000]}

Genera preguntas específicas y útiles que ayuden a explorar los aspectos clave del documento.
Lista las preguntas numeradas del 1 al {num_questions}.
"""

        try:
            response = self.vertex_service.generate_text(
                prompt,
                max_output_tokens=65535  # Maximum for Gemini 2.5 Pro (includes thinking tokens)
            )

            # Parse questions from response
            questions = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering
                    question = line.split('.', 1)[-1].strip()
                    if question:
                        questions.append(question)

            return questions[:num_questions]

        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return []
