"""Text chunking utilities."""
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class TextChunker:
    """Handles text chunking for embeddings and processing."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize text chunker.

        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Chunk text into overlapping segments.

        Args:
            text: Input text
            metadata: Optional metadata to attach to chunks

        Returns:
            List of chunk dictionaries
        """
        if not text:
            return []

        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            # Extract chunk
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)

                if break_point > self.chunk_size * 0.7:  # At least 70% of chunk size
                    chunk_text = chunk_text[:break_point + 1]
                    end = start + break_point + 1

            # Create chunk dictionary
            chunk = {
                'id': chunk_id,
                'content': chunk_text.strip(),
                'start_pos': start,
                'end_pos': end,
                'size': len(chunk_text)
            }

            # Add metadata if provided
            if metadata:
                chunk.update(metadata)

            chunks.append(chunk)

            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            chunk_id += 1

        logger.info(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks

    def chunk_by_sections(self,
                         sections: List[Dict],
                         pages_text: Dict[int, str]) -> List[Dict]:
        """Chunk text organized by document sections.

        Args:
            sections: List of section definitions
            pages_text: Dictionary mapping page numbers to text

        Returns:
            List of chunks with section metadata
        """
        all_chunks = []

        for section in sections:
            section_title = section.get('title', 'Unknown')
            start_page = section.get('start_page', 0)
            end_page = section.get('end_page', start_page + 1)

            logger.info(f"Chunking section: {section_title} (pages {start_page}-{end_page})")

            # Collect text for this section
            section_text = []
            for page_num in range(start_page, end_page):
                if page_num in pages_text:
                    section_text.append(pages_text[page_num])

            combined_text = "\n\n".join(section_text)

            # Chunk the section text
            metadata = {
                'section': section_title,
                'section_level': section.get('level', 1),
                'start_page': start_page,
                'end_page': end_page
            }

            section_chunks = self.chunk_text(combined_text, metadata)
            all_chunks.extend(section_chunks)

        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks

    def chunk_with_context(self,
                          text: str,
                          context_window: int = 100,
                          metadata: Dict = None) -> List[Dict]:
        """Chunk text with surrounding context.

        Args:
            text: Input text
            context_window: Characters of context to include
            metadata: Optional metadata

        Returns:
            List of chunks with context
        """
        base_chunks = self.chunk_text(text, metadata)

        # Add context to each chunk
        for chunk in base_chunks:
            start = chunk['start_pos']
            end = chunk['end_pos']

            # Get preceding context
            context_start = max(0, start - context_window)
            preceding_context = text[context_start:start]

            # Get following context
            context_end = min(len(text), end + context_window)
            following_context = text[end:context_end]

            chunk['preceding_context'] = preceding_context
            chunk['following_context'] = following_context
            chunk['full_context'] = (
                preceding_context + chunk['content'] + following_context
            )

        return base_chunks

    def merge_small_chunks(self,
                          chunks: List[Dict],
                          min_size: int = 200) -> List[Dict]:
        """Merge chunks that are too small.

        Args:
            chunks: List of chunks
            min_size: Minimum chunk size

        Returns:
            List of merged chunks
        """
        if not chunks:
            return []

        merged = []
        current_chunk = None

        for chunk in chunks:
            if current_chunk is None:
                current_chunk = chunk.copy()
            elif chunk['size'] < min_size:
                # Merge with previous chunk
                current_chunk['content'] += ' ' + chunk['content']
                current_chunk['end_pos'] = chunk['end_pos']
                current_chunk['size'] = len(current_chunk['content'])
            else:
                merged.append(current_chunk)
                current_chunk = chunk.copy()

        # Add last chunk
        if current_chunk:
            merged.append(current_chunk)

        logger.info(f"Merged {len(chunks)} chunks into {len(merged)} chunks")
        return merged
