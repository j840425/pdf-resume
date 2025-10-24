"""PDF processing and text extraction."""
import PyPDF2
import pdfplumber
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF text extraction and metadata."""

    def __init__(self, max_pages: int = 2000):
        """Initialize PDF processor.

        Args:
            max_pages: Maximum number of pages to process
        """
        self.max_pages = max_pages

    def validate_pdf(self, pdf_path: Path) -> Dict[str, any]:
        """Validate PDF and return metadata.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with validation results and metadata
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = len(reader.pages)

                if num_pages > self.max_pages:
                    return {
                        "valid": False,
                        "error": f"PDF exceeds maximum pages ({num_pages} > {self.max_pages})",
                        "num_pages": num_pages
                    }

                metadata = reader.metadata
                return {
                    "valid": True,
                    "num_pages": num_pages,
                    "title": metadata.get("/Title", "Unknown") if metadata else "Unknown",
                    "author": metadata.get("/Author", "Unknown") if metadata else "Unknown"
                }
        except Exception as e:
            logger.error(f"Error validating PDF: {e}")
            return {
                "valid": False,
                "error": str(e)
            }

    def extract_text(self, pdf_path: Path) -> str:
        """Extract all text from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text content
        """
        text_content = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_content.append(text)

            return "\n\n".join(text_content)
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            raise

    def extract_text_by_pages(self, pdf_path: Path,
                              start_page: int = 0,
                              end_page: Optional[int] = None) -> Dict[int, str]:
        """Extract text from specific page range.

        Args:
            pdf_path: Path to PDF file
            start_page: Starting page (0-indexed)
            end_page: Ending page (exclusive), None for all pages

        Returns:
            Dictionary mapping page numbers to text content
        """
        pages_text = {}

        try:
            with pdfplumber.open(pdf_path) as pdf:
                end = end_page if end_page else len(pdf.pages)

                for i in range(start_page, min(end, len(pdf.pages))):
                    text = pdf.pages[i].extract_text()
                    if text:
                        pages_text[i] = text

            return pages_text
        except Exception as e:
            logger.error(f"Error extracting pages {start_page}-{end_page}: {e}")
            raise

    def extract_table_of_contents(self, pdf_path: Path) -> Optional[List[Dict]]:
        """Attempt to extract table of contents from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of TOC entries with title and page number, or None
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)

                # Try to get outlines (bookmarks)
                if hasattr(reader, 'outline') and reader.outline:
                    toc = []
                    self._parse_outline(reader.outline, toc)
                    return toc if toc else None

            return None
        except Exception as e:
            logger.warning(f"Could not extract TOC: {e}")
            return None

    def _parse_outline(self, outline: List, toc: List, level: int = 0):
        """Recursively parse PDF outline structure.

        Args:
            outline: PDF outline structure
            toc: List to append TOC entries to
            level: Current nesting level
        """
        for item in outline:
            if isinstance(item, list):
                self._parse_outline(item, toc, level + 1)
            else:
                try:
                    title = item.get('/Title', 'Unknown')
                    # Get page number from destination
                    page_num = None
                    if '/Page' in item:
                        page_num = item['/Page']

                    toc.append({
                        'title': title,
                        'page': page_num,
                        'level': level
                    })
                except Exception as e:
                    logger.debug(f"Error parsing outline item: {e}")
                    continue
