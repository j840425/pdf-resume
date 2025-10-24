"""Document structure detection using LLM."""
import logging
from typing import Dict, List, Optional, Tuple
import json
import re

logger = logging.getLogger(__name__)


class StructureDetector:
    """Detects document structure using Vertex AI Gemini."""

    def __init__(self, vertex_service):
        """Initialize structure detector.

        Args:
            vertex_service: VertexService instance
        """
        self.vertex_service = vertex_service
        self.max_toc_search_pages = 30  # Máximo de páginas a buscar el índice

    def detect_structure(self,
                        text_sample: str,
                        toc: Optional[List[Dict]] = None,
                        num_pages: int = 0,
                        pages_text: Optional[Dict[int, str]] = None) -> Dict:
        """Detect document structure from text and TOC.

        Args:
            text_sample: Sample text from document (first pages)
            toc: Table of contents if available
            num_pages: Total number of pages
            pages_text: Full pages text dictionary for better analysis

        Returns:
            Dictionary with detected sections and structure
        """
        logger.info("Detecting document structure...")

        try:
            # Step 1: Find TOC/Index page using LLM
            toc_location = self._find_toc_page(pages_text, num_pages)

            if toc_location['found']:
                logger.info(f"TOC found at pages {toc_location['start_page']}-{toc_location['end_page']}")

                # Step 2: Extract structure from TOC using LLM
                structure = self._extract_structure_from_toc(
                    pages_text,
                    toc_location,
                    num_pages
                )
            else:
                logger.warning("TOC not found, using heuristic analysis")
                # Fallback: analyze document structure without TOC
                structure = self._analyze_structure_without_toc(
                    pages_text,
                    num_pages
                )

            # Step 3: Validate and refine sections
            structure = self._validate_and_refine_structure(structure, num_pages)

            logger.info(f"Detected {len(structure.get('sections', []))} sections")
            return structure

        except Exception as e:
            logger.error(f"Error detecting structure: {e}", exc_info=True)
            # Return fallback structure
            return self._create_fallback_structure(num_pages)

    def _build_structure_prompt(self,
                                text_sample: str,
                                toc: Optional[List[Dict]],
                                num_pages: int) -> str:
        """Build prompt for structure detection.

        Args:
            text_sample: Sample text
            toc: Table of contents
            num_pages: Total pages

        Returns:
            Formatted prompt
        """
        prompt = f"""Analiza la siguiente información de un documento PDF y detecta su estructura organizacional.

INFORMACIÓN DEL DOCUMENTO:
- Total de páginas: {num_pages}

"""

        if toc:
            prompt += "TABLA DE CONTENIDOS:\n"
            for item in toc[:20]:  # Limit to first 20 entries
                prompt += f"- {item.get('title', 'Unknown')} (Página {item.get('page', '?')})\n"
            prompt += "\n"

        prompt += f"""MUESTRA DE TEXTO (Primeras páginas):
{text_sample[:3000]}

TAREA:
Identifica las secciones principales del documento y organízalas de forma jerárquica.
Para cada sección, proporciona:
1. Título de la sección
2. Página de inicio (aproximada si no está en el TOC)
3. Página final (aproximada)
4. Nivel jerárquico (1 = principal, 2 = subsección, etc.)
5. Descripción breve del contenido

Devuelve la respuesta en formato JSON con esta estructura:
{{
  "sections": [
    {{
      "title": "Nombre de la sección",
      "start_page": 1,
      "end_page": 10,
      "level": 1,
      "description": "Descripción breve"
    }}
  ]
}}
"""
        return prompt

    def _parse_structure_response(self, response: str) -> Dict:
        """Parse LLM response to structured format.

        Args:
            response: LLM response text

        Returns:
            Parsed structure dictionary
        """
        try:
            # Remove markdown code blocks if present
            response_clean = response.replace('```json', '').replace('```', '').strip()

            # Try to find JSON in response
            start_idx = response_clean.find('{')
            end_idx = response_clean.rfind('}') + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_clean[start_idx:end_idx]

                # Remove trailing commas (common JSON error from LLMs)
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

                structure = json.loads(json_str)
                return structure
            else:
                logger.warning("No JSON found in response")
                return {"sections": []}

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            logger.debug(f"Failed JSON: {json_str[:200] if 'json_str' in locals() else 'N/A'}...")
            return {"sections": []}

    def _create_fallback_structure(self, num_pages: int) -> Dict:
        """Create fallback structure when detection fails.

        Args:
            num_pages: Total number of pages

        Returns:
            Basic structure dictionary
        """
        pages_per_section = 50
        sections = []

        for i in range(0, num_pages, pages_per_section):
            end_page = min(i + pages_per_section, num_pages)
            sections.append({
                "title": f"Sección {i // pages_per_section + 1}",
                "start_page": i,
                "end_page": end_page,
                "level": 1,
                "description": f"Páginas {i} a {end_page}"
            })

        return {"sections": sections}

    def refine_sections_with_content(self,
                                     sections: List[Dict],
                                     pages_text: Dict[int, str]) -> List[Dict]:
        """Refine section boundaries based on actual content.

        Args:
            sections: Initial sections from structure detection
            pages_text: Dictionary of page numbers to text content

        Returns:
            Refined sections list
        """
        # This could use additional LLM calls to verify section boundaries
        # For now, return sections as-is
        return sections

    def _find_toc_page(self, pages_text: Dict[int, str], num_pages: int) -> Dict:
        """Find the table of contents page using LLM page by page analysis.

        Args:
            pages_text: Dictionary of page numbers to text
            num_pages: Total number of pages

        Returns:
            Dictionary with TOC location info
        """
        logger.info("Searching for table of contents...")

        if not pages_text:
            return {'found': False}

        # Search in first N pages
        max_search = min(self.max_toc_search_pages, num_pages)

        # Analyze pages in batches for efficiency
        batch_size = 3
        for start_idx in range(0, max_search, batch_size):
            end_idx = min(start_idx + batch_size, max_search)

            # Collect pages for batch
            batch_pages = []
            for page_num in range(start_idx, end_idx):
                if page_num in pages_text:
                    page_text = pages_text[page_num][:2000]  # First 2000 chars
                    batch_pages.append({
                        'page_num': page_num,
                        'text': page_text
                    })

            if not batch_pages:
                continue

            # Log batch being analyzed
            page_nums = [p['page_num'] for p in batch_pages]
            logger.info(f"Analyzing batch: pages {page_nums}")

            # Log preview of pages in batch
            for p in batch_pages:
                preview = p['text'][:150].replace('\n', ' ')
                logger.debug(f"  Page {p['page_num']} preview: {preview}...")

            # Ask LLM to identify TOC
            result = self._analyze_pages_for_toc(batch_pages)
            logger.info(f"Batch {page_nums} result: found={result.get('found', False)}")

            if result['found']:
                # Verify and expand TOC range
                page_num = result['page_num']

                # Handle case where LLM returns a list instead of single number
                if isinstance(page_num, list):
                    toc_start = page_num[0] if page_num else start_idx
                    toc_end = page_num[-1] if len(page_num) > 1 else toc_start
                else:
                    toc_start = page_num
                    toc_end = self._find_toc_end_page(pages_text, toc_start, max_search)

                return {
                    'found': True,
                    'start_page': toc_start,
                    'end_page': toc_end,
                    'confidence': result.get('confidence', 'high')
                }

        logger.info("Table of contents not found in first pages")
        return {'found': False}

    def _analyze_pages_for_toc(self, batch_pages: List[Dict]) -> Dict:
        """Analyze a batch of pages to identify TOC.

        Args:
            batch_pages: List of page dictionaries with page_num and text

        Returns:
            Dictionary with found status and page number
        """
        prompt = self._build_toc_detection_prompt(batch_pages)

        try:
            response = self.vertex_service.generate_text(
                prompt,
                temperature=0.1,  # Low temperature for factual task
                max_output_tokens=4096  # Sufficient for TOC detection JSON
            )

            logger.debug(f"TOC detection LLM response: {response[:500]}")

            result = self._parse_toc_detection_response(response, batch_pages)

            if result.get('found'):
                logger.info(f"✓ TOC detected! Page: {result.get('page_num')}, Reasoning: {result.get('reasoning', 'N/A')[:100]}")

            return result

        except Exception as e:
            logger.error(f"Error analyzing pages for TOC: {e}")
            return {'found': False}

    def _build_toc_detection_prompt(self, batch_pages: List[Dict]) -> str:
        """Build prompt for TOC detection.

        Args:
            batch_pages: List of page dictionaries

        Returns:
            Formatted prompt
        """
        prompt = """Analiza las siguientes páginas de un documento PDF para identificar si alguna contiene el INICIO de una TABLA DE CONTENIDOS (TOC), ÍNDICE, o ÍNDICE DE CONTENIDO.

Una tabla de contenidos típicamente contiene:
- Títulos de capítulos o secciones
- Números de página correspondientes (pueden aparecer al lado, debajo, o cerca de los títulos)
- Estructura jerárquica o numeración (CHAPTER 1, CHAPTER 2, etc.)
- Términos como: "Contenido", "Índice", "Table of Contents", "Contents", "Índice General"
- Lista de múltiples secciones/capítulos del documento

IMPORTANTE: Un TOC puede tener formatos variados:
- Títulos con palabras concatenadas sin espacios (ej: "WhoCanBecomeaQuantitativeTrader?")
- Números de página separados de los títulos
- Encabezados técnicos de impresión que debes IGNORAR (ej: "P1:JYS fm JWBK321-Chan")
- Numeración romana al final de página (vii, viii, xi, xvii) es NORMAL en TOCs y NO debe ser motivo para descartarla
- El texto "Preface", "Acknowledgments", "CHAPTER" son indicadores FUERTES de un TOC

CRITERIO CRÍTICO PARA DETECTAR EL INICIO DEL TOC:
- Si ves "Contents" como título en una página, incluso si solo muestra 1-2 capítulos, es MUY PROBABLE que sea el INICIO del TOC
- La primera aparición de "Contents" o "Índice" como título principal indica el INICIO del TOC
- NO esperes a ver la página completa con todos los capítulos - detecta el INICIO

PÁGINAS A ANALIZAR:
"""

        for page_info in batch_pages:
            prompt += f"\n{'='*60}\n"
            prompt += f"PÁGINA {page_info['page_num']}:\n"
            prompt += f"{page_info['text']}\n"

        prompt += f"""
{'='*60}

INSTRUCCIONES:
1. Analiza CUIDADOSAMENTE cada página
2. Identifica si alguna página contiene el INICIO de una tabla de contenidos
3. BUSCA ESPECÍFICAMENTE:
   - La palabra "Contents", "Contenido", "Índice" como título principal de la página
   - Presencia de "Preface", "Acknowledgments" al inicio de la lista
   - Múltiples líneas con títulos de capítulos (CHAPTER 1, CHAPTER 2, etc.)
   - Patrones donde aparecen títulos seguidos de números (números de página)
4. IGNORA encabezados técnicos del tipo "P1:JYS fm JWBK321-Chan" o "Printer:Yettocome"
5. IGNORA numeración romana sola al final (vii, viii, xi, xvii) - es numeración de página del TOC
6. Una TOC puede tener títulos sin espacios entre palabras (formato compacto)

REGLA FUNDAMENTAL:
Si una página tiene el título "Contents" o "Índice" Y muestra al menos 1-2 capítulos o secciones,
DEBES marcarla como found: true, incluso si la lista parece incompleta.

RESPONDE EN FORMATO JSON ESTRICTO:
{{
  "found": true/false,
  "page_num": <UN SOLO número de página donde INICIA la TOC, o null>,
  "confidence": "high/medium/low",
  "reasoning": "Breve explicación"
}}

EJEMPLOS DE INICIO DE TOC VÁLIDOS:
✅ PÁGINA con título "Contents" + "Preface xi" + "Acknowledgments xvii" + "CHAPTER 1... 1"
✅ PÁGINA con "Contents" + "CHAPTER 1" + "CHAPTER 2" (incluso si solo hay 2 capítulos)
✅ PÁGINA con "Índice" + lista de secciones con números de página

EJEMPLOS DE NO TOC:
❌ Página de dedicatoria ("To my parents...")
❌ Página en blanco o solo con numeración romana
❌ Texto de capítulo regular sin lista estructurada

REGLAS CRÍTICAS:
- Busca el INICIO del TOC, no la continuación
- Si ves "Contents" como título principal, es casi seguro el INICIO del TOC
- page_num debe ser la PRIMERA página donde aparece "Contents" o "Índice" como título
- NO confundas "viii CONTENTS" (continuación) con "Contents" al inicio de la página (inicio)
"""
        return prompt

    def _parse_toc_detection_response(self, response: str, batch_pages: List[Dict]) -> Dict:
        """Parse LLM response for TOC detection.

        Args:
            response: LLM response
            batch_pages: Original batch pages for context

        Returns:
            Parsed result dictionary
        """
        try:
            # Remove markdown code blocks if present
            response_clean = response.replace('```json', '').replace('```', '').strip()

            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_clean, re.DOTALL)

            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)

                if result.get('found') and result.get('page_num') is not None:
                    logger.info(f"TOC detected at page {result['page_num']}: {result.get('reasoning', '')}")
                    return result

            return {'found': False}

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing TOC detection response: {e}")
            return {'found': False}

    def _find_toc_end_page(self, pages_text: Dict[int, str], start_page: int, max_page: int) -> int:
        """Find where the TOC ends.

        Args:
            pages_text: Dictionary of pages
            start_page: TOC start page
            max_page: Maximum page to search

        Returns:
            End page number
        """
        logger.info(f"Finding TOC end page starting from page {start_page}")
        last_toc_page = start_page

        # Check next few pages to see if TOC continues
        for page_num in range(start_page + 1, min(start_page + 10, max_page)):
            if page_num not in pages_text:
                logger.debug(f"Page {page_num} not in pages_text, skipping")
                continue

            page_preview = pages_text[page_num][:500]
            logger.debug(f"Checking page {page_num} for TOC continuation. Preview: {page_preview[:100]}")

            # Use LLM to check if page is still TOC
            prompt = f"""Determina si la siguiente página es una CONTINUACIÓN de una tabla de contenidos.

Una CONTINUACIÓN de TOC típicamente contiene:
- Título "CONTENTS" o "Contents" con numeración romana (ej: "viii CONTENTS", "ix Contents")
- Lista de capítulos (CHAPTER 3, CHAPTER 4, etc.) con números de página
- Subsecciones con números de página
- Estructura similar a la página anterior del TOC

NO es continuación si:
- Es una página en blanco o solo con numeración romana
- Comienza el texto de un capítulo (ej: "This book is about...", texto narrativo)
- Es el Preface o texto introductorio

PÁGINA {page_num}:
{pages_text[page_num][:2000]}

Responde SOLO con JSON:
{{
  "is_toc_continuation": true/false,
  "reasoning": "breve explicación"
}}
"""

            try:
                response = self.vertex_service.generate_text(prompt, temperature=0.1, max_output_tokens=8192)  # Increased for thinking tokens

                logger.debug(f"TOC continuation check for page {page_num}: {response[:200]}")

                # Remove markdown code blocks if present
                response_clean = response.replace('```json', '').replace('```', '').strip()

                # First check if it's incomplete JSON with just the boolean value
                incomplete_match = re.search(r'\{\s*"is_toc_continuation"\s*:\s*(true|false)', response_clean, re.DOTALL | re.IGNORECASE)

                if incomplete_match:
                    is_cont = incomplete_match.group(1).lower() == 'true'

                    # Try to parse as complete JSON first
                    try:
                        # Try to extract complete JSON
                        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_clean, re.DOTALL)
                        if not json_match:
                            json_match = re.search(r'\{.*?\}', response_clean, re.DOTALL)

                        if json_match:
                            result = json.loads(json_match.group(0))
                            is_continuation = result.get('is_toc_continuation', False)
                            reasoning = result.get('reasoning', '')
                            logger.info(f"Page {page_num} is_toc_continuation: {is_continuation} - {reasoning}")
                        else:
                            # JSON incomplete, use extracted value
                            is_continuation = is_cont
                            reasoning = 'Response was incomplete'
                            logger.info(f"Page {page_num} is_toc_continuation: {is_continuation} - {reasoning} (incomplete JSON)")

                    except json.JSONDecodeError:
                        # JSON is malformed, use extracted boolean value
                        is_continuation = is_cont
                        reasoning = 'JSON was malformed, extracted boolean'
                        logger.info(f"Page {page_num} is_toc_continuation: {is_continuation} - {reasoning} (malformed JSON)")

                    # Process result
                    if is_continuation:
                        last_toc_page = page_num
                    else:
                        # Found first non-TOC page
                        logger.info(f"TOC ends at page {last_toc_page}")
                        return last_toc_page
                else:
                    logger.warning(f"No is_toc_continuation found in response for page {page_num}")
                    logger.warning(f"Full response: {response[:500]}")

            except Exception as e:
                logger.warning(f"Error checking TOC continuation for page {page_num}: {e}")
                # Continue checking next pages instead of returning early

        logger.info(f"Reached max search, TOC ends at page {last_toc_page}")
        return last_toc_page

    def _extract_structure_from_toc(self,
                                    pages_text: Dict[int, str],
                                    toc_location: Dict,
                                    num_pages: int) -> Dict:
        """Extract document structure from detected TOC.

        Args:
            pages_text: All pages text
            toc_location: TOC location info
            num_pages: Total pages

        Returns:
            Structure dictionary with sections
        """
        logger.info("Extracting structure from TOC...")

        # Collect all TOC pages
        toc_text = ""
        for page_num in range(toc_location['start_page'], toc_location['end_page'] + 1):
            if page_num in pages_text:
                toc_text += f"\n--- Página {page_num} ---\n"
                toc_text += pages_text[page_num]

        logger.info(f"TOC text length: {len(toc_text)} characters")
        logger.info(f"TOC preview (first 500 chars):\n{toc_text[:500]}")

        # Use LLM to parse TOC into structured sections
        prompt = f"""Extrae TODA la estructura del documento de la siguiente tabla de contenidos.

TABLA DE CONTENIDOS COMPLETA:
{toc_text}

INFORMACIÓN:
- Total de páginas del documento: {num_pages}
- Debes extraer TODOS los capítulos Y TODAS sus subsecciones

TAREA CRÍTICA:
Convierte esta TOC en una lista estructurada de secciones con JERARQUÍA CLARA:

1. Identifica el título exacto de cada entrada
2. Extrae el número de página de inicio
3. **CRÍTICO**: Determina el nivel jerárquico correctamente:
   - **Nivel 1**: Capítulos principales (CHAPTER, PART, SECTION principal)
   - **Nivel 2**: Subsecciones dentro de capítulos (1.1, 1.2, subcapítulos)
   - **Nivel 3**: Sub-subsecciones (1.1.1, 1.1.2)
4. Calcula la página de fin basándote en la siguiente sección del MISMO NIVEL o superior

REGLAS CRÍTICAS PARA JERARQUÍA:
- **IMPORTANTE**: Identifica y extrae TODAS las entradas del TOC, no solo algunas
- Identifica los CAPÍTULOS PRINCIPALES (nivel 1) - generalmente tienen títulos como "CHAPTER X", "PART X", o son secciones principales
- Identifica TODAS las SUBSECCIONES (nivel 2) que están dentro de un capítulo - tienen numeración como "1.1", "1.2" o son sub-temas indentados
- Si hay numeración (1.1, 2.3, etc.), cuenta los puntos para determinar el nivel (1.1 = nivel 2, 1.1.1 = nivel 3)
- Si NO hay numeración, usa la indentación o el contexto para determinar la jerarquía
- **CRÍTICO**: Incluye ABSOLUTAMENTE TODAS las secciones que aparezcan en el TOC con números de página
- Si el TOC tiene 50 entradas, tu JSON debe tener 50 elementos en el array "sections"
- Las páginas deben ser números válidos entre 1 y {num_pages}
- TODOS los valores de texto deben estar entre comillas dobles
- NO uses comillas simples
- NO pongas coma después del último elemento de un array
- section_number debe contener la numeración si existe (ej: "1.2", "2.3.1") o "" si no existe

FORMATO JSON REQUERIDO (respeta EXACTAMENTE este formato):
{{
  "sections": [
    {{
      "title": "Example Title",
      "start_page": 10,
      "end_page": 25,
      "level": 1,
      "section_number": ""
    }},
    {{
      "title": "Another Section",
      "start_page": 26,
      "end_page": 50,
      "level": 1,
      "section_number": ""
    }}
  ]
}}

INSTRUCCIONES FINALES:
1. Devuelve ÚNICAMENTE el JSON, sin texto adicional antes o después
2. Extrae TODAS las secciones del TOC - no te detengas a la mitad
3. Si el TOC tiene muchas entradas, asegúrate de incluirlas TODAS en el JSON
4. Verifica que el JSON sea válido antes de devolver
5. NO uses comas trailing (coma antes de }} o ])
6. Asegúrate de que start_page < end_page para cada sección

CUENTA FINAL: Antes de devolver el JSON, cuenta cuántas entradas tiene el TOC y verifica que tu JSON tenga la misma cantidad en el array "sections".
"""

        try:
            # Gemini 2.5 Pro uses thinking tokens that count against max_output_tokens
            # Set very high limit to compensate for internal thinking budget
            response = self.vertex_service.generate_text(
                prompt,
                temperature=0.0,  # Zero temperature for deterministic JSON generation
                max_output_tokens=65535  # Maximum allowed for Gemini 2.5 Pro
            )

            # Log the full response for debugging
            logger.info("="*60)
            logger.info("LLM RESPONSE FOR TOC EXTRACTION:")
            logger.info(f"Response length: {len(response)} characters")
            logger.info(response[:2000])  # First 2000 chars
            if len(response) > 2000:
                logger.info(f"... (truncated, {len(response) - 2000} more chars)")
            logger.info("="*60)

            structure = self._parse_structure_response(response)

            if not structure.get('sections'):
                logger.warning("Failed to extract sections from TOC, using fallback")
                logger.warning(f"Full LLM response: {response[:500]}")
                return self._create_fallback_structure(num_pages)

            logger.info(f"Successfully extracted {len(structure['sections'])} sections from TOC")
            for i, section in enumerate(structure['sections'][:5]):  # Log first 5
                logger.info(f"  Section {i+1}: {section.get('title')} (level {section.get('level')})")

            return structure

        except Exception as e:
            logger.error(f"Error extracting structure from TOC: {e}")
            logger.error(f"Full response: {response[:500] if 'response' in locals() else 'No response'}")
            return self._create_fallback_structure(num_pages)

    def _analyze_structure_without_toc(self,
                                       pages_text: Dict[int, str],
                                       num_pages: int) -> Dict:
        """Analyze document structure when no TOC is found.

        Args:
            pages_text: All pages text
            num_pages: Total pages

        Returns:
            Structure dictionary
        """
        logger.info("Analyzing structure without TOC using content analysis...")

        # Sample pages throughout the document
        sample_pages = []
        step = max(1, num_pages // 20)  # Sample ~20 pages

        for page_num in range(0, num_pages, step):
            if page_num in pages_text:
                sample_pages.append({
                    'page_num': page_num,
                    'text': pages_text[page_num][:1000]
                })

        if len(sample_pages) < 3:
            return self._create_fallback_structure(num_pages)

        # Use LLM to identify section patterns
        prompt = f"""Analiza las siguientes muestras de un documento de {num_pages} páginas e identifica su estructura.

MUESTRAS DE PÁGINAS:
"""

        for sample in sample_pages[:15]:  # Limit to avoid token limit
            prompt += f"\n{'='*50}\nPÁGINA {sample['page_num']}:\n{sample['text']}\n"

        prompt += f"""
{'='*50}

TAREA:
Identifica los principales capítulos o secciones del documento basándote en:
- Títulos en fuente grande o negritas
- Numeración de capítulos (Capítulo 1, Chapter 1, 1., etc.)
- Cambios temáticos importantes
- Patrones de encabezados

Estima las páginas de inicio/fin de cada sección basándote en las muestras.

RESPONDE EN JSON:
{{
  "sections": [
    {{
      "title": "Título de la sección",
      "start_page": <número estimado>,
      "end_page": <número estimado>,
      "level": 1,
      "confidence": "high/medium/low"
    }}
  ]
}}

Devuelve al menos 3-5 secciones principales que cubran todo el documento.
"""

        try:
            response = self.vertex_service.generate_text(
                prompt,
                temperature=0.3,
                max_output_tokens=65535  # Maximum for complex structure analysis without TOC
            )

            structure = self._parse_structure_response(response)

            if not structure.get('sections'):
                return self._create_fallback_structure(num_pages)

            return structure

        except Exception as e:
            logger.error(f"Error analyzing structure: {e}")
            return self._create_fallback_structure(num_pages)

    def _validate_and_refine_structure(self, structure: Dict, num_pages: int) -> Dict:
        """Validate and refine the detected structure.

        Args:
            structure: Initial structure
            num_pages: Total pages

        Returns:
            Refined structure
        """
        sections = structure.get('sections', [])

        if not sections:
            return structure

        refined_sections = []

        for i, section in enumerate(sections):
            # Validate page numbers
            start = max(0, min(section.get('start_page', 0), num_pages))
            end = section.get('end_page', num_pages)

            # If end_page is missing or invalid, estimate from next section
            if i < len(sections) - 1:
                next_start = sections[i + 1].get('start_page', num_pages)
                end = min(end, next_start - 1)
            else:
                end = num_pages

            end = max(start, min(end, num_pages))

            # Skip invalid sections
            if start >= end or start >= num_pages:
                logger.warning(f"Skipping invalid section: {section.get('title', 'Unknown')}")
                continue

            refined_sections.append({
                'title': section.get('title', f'Section {i+1}'),
                'start_page': start,
                'end_page': end,
                'level': section.get('level', 1),
                'section_number': section.get('section_number', ''),
                'description': section.get('description', '')
            })

        structure['sections'] = refined_sections
        return structure

    def get_summary_units(self, structure: Dict) -> List[Dict]:
        """Get the appropriate units for summarization (level 2, or level 1 if no level 2 exists).

        Args:
            structure: Document structure with sections

        Returns:
            List of sections to be summarized
        """
        sections = structure.get('sections', [])

        if not sections:
            return []

        # Analyze hierarchy
        has_level_2 = any(s.get('level', 1) == 2 for s in sections)

        if has_level_2:
            # Summarize at level 2 (subsections)
            summary_units = [s for s in sections if s.get('level', 1) == 2]
            logger.info(f"Using level 2 subsections for summaries: {len(summary_units)} units")
        else:
            # Fallback: summarize level 1 (chapters)
            summary_units = [s for s in sections if s.get('level', 1) == 1]
            logger.info(f"No level 2 found, using level 1 chapters for summaries: {len(summary_units)} units")

        return summary_units

    def get_hierarchical_structure(self, structure: Dict) -> Dict:
        """Organize sections into hierarchical structure for display.

        Args:
            structure: Flat structure with sections

        Returns:
            Hierarchical structure with chapters and subsections
        """
        sections = structure.get('sections', [])
        hierarchical = {
            'chapters': []
        }

        current_chapter = None

        for section in sections:
            level = section.get('level', 1)

            if level == 1:
                # New chapter
                current_chapter = {
                    **section,
                    'subsections': []
                }
                hierarchical['chapters'].append(current_chapter)
            elif level == 2 and current_chapter:
                # Subsection of current chapter
                current_chapter['subsections'].append(section)
            elif level >= 3 and current_chapter and current_chapter['subsections']:
                # Sub-subsection - attach to last subsection
                current_chapter['subsections'][-1].setdefault('sub_subsections', []).append(section)

        return hierarchical
