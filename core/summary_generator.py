"""Summary generation using Gemini."""
import logging
from typing import Dict, List
from config import DetailLevel

logger = logging.getLogger(__name__)


class SummaryGenerator:
    """Generates summaries for document sections."""

    def __init__(self, vertex_service):
        """Initialize summary generator.

        Args:
            vertex_service: VertexService instance
        """
        self.vertex_service = vertex_service

    def generate_section_summary(self,
                                 section: Dict,
                                 content: str,
                                 detail_level: DetailLevel) -> str:
        """Generate summary for a document section.

        Args:
            section: Section metadata (title, pages, etc.)
            content: Text content of the section
            detail_level: Level of detail for summary

        Returns:
            Generated summary text
        """
        logger.info(f"Generating {detail_level} summary for: {section.get('title')}")

        prompt = self._build_summary_prompt(section, content, detail_level)

        try:
            summary = self.vertex_service.generate_text(
                prompt,
                temperature=0.2,  # Low temperature for consistent, focused summaries
                max_output_tokens=65535  # Maximum for Gemini 2.5 Pro (includes thinking tokens)
            )
            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Error al generar resumen: {str(e)}"

    def generate_document_summary(self,
                                  sections_summaries: List[Dict],
                                  detail_level: DetailLevel) -> str:
        """Generate overall document summary from section summaries.

        Args:
            sections_summaries: List of section summaries
            detail_level: Level of detail

        Returns:
            Overall document summary
        """
        logger.info(f"Generating overall document summary ({detail_level})")

        # Combine section summaries
        combined_text = "\n\n".join([
            f"## {s.get('section', {}).get('title', 'Sin título')}\n{s.get('summary', '')}"
            for s in sections_summaries
        ])

        prompt = f"""Genera un resumen general del siguiente documento basándote en los resúmenes de sus secciones.

Nivel de detalle: {detail_level.value}

RESÚMENES POR SECCIÓN:
{combined_text[:5000]}

Proporciona un resumen coherente que capture los puntos principales del documento completo.
"""

        try:
            summary = self.vertex_service.generate_text(
                prompt,
                temperature=0.2,  # Low temperature for consistent, focused summaries
                max_output_tokens=65535  # Maximum for Gemini 2.5 Pro (includes thinking tokens)
            )
            return summary

        except Exception as e:
            logger.error(f"Error generating document summary: {e}")
            return "Error al generar resumen general del documento."

    def _build_summary_prompt(self,
                             section: Dict,
                             content: str,
                             detail_level: DetailLevel) -> str:
        """Build prompt for summary generation.

        Args:
            section: Section metadata
            content: Section content
            detail_level: Detail level

        Returns:
            Formatted prompt
        """
        detail_instructions = {
            DetailLevel.EXECUTIVE: """
- Máximo 2-3 párrafos
- Solo los puntos más críticos
- Enfoque en conclusiones y recomendaciones clave
- Lenguaje conciso para ejecutivos
""",
            DetailLevel.NORMAL: """
- 3-5 párrafos
- Puntos principales y secundarios relevantes
- Balance entre detalle y concisión
- Incluir datos importantes
""",
            DetailLevel.DETAILED: """
- Resumen exhaustivo (5-8 párrafos o más)
- Incluir todos los puntos importantes
- Detalles técnicos, datos, estadísticas
- Ejemplos y casos específicos mencionados
- Mantener estructura lógica del contenido original
"""
        }

        prompt = f"""Genera un resumen de la siguiente sección del documento.

SECCIÓN: {section.get('title', 'Sin título')}
PÁGINAS: {section.get('start_page')} - {section.get('end_page')}
NIVEL DE DETALLE: {detail_level.value}

INSTRUCCIONES PARA NIVEL {detail_level.value.upper()}:
{detail_instructions.get(detail_level, detail_instructions[DetailLevel.NORMAL])}

CONTENIDO DE LA SECCIÓN:
{content[:8000]}

Genera el resumen en español, con estructura clara y puntos bien organizados.
"""

        return prompt

    def generate_batch_summaries(self,
                                sections_with_content: List[Dict],
                                detail_level: DetailLevel) -> List[Dict]:
        """Generate summaries for multiple sections.

        Args:
            sections_with_content: List of sections with their content
            detail_level: Detail level for all summaries

        Returns:
            List of sections with added summaries
        """
        results = []

        for idx, section_data in enumerate(sections_with_content):
            logger.info(f"Processing section {idx + 1}/{len(sections_with_content)}")

            section = section_data.get('section', {})
            content = section_data.get('content', '')

            summary = self.generate_section_summary(
                section,
                content,
                detail_level
            )

            results.append({
                **section_data,
                'summary': summary
            })

        return results
