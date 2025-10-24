"""Input validation utilities."""
import logging
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class PDFValidator:
    """Validates PDF files and inputs."""

    def __init__(self, max_file_size_mb: int = 100, max_pages: int = 2000):
        """Initialize validator.

        Args:
            max_file_size_mb: Maximum file size in MB
            max_pages: Maximum number of pages
        """
        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert to bytes
        self.max_pages = max_pages

    def validate_file(self, file_path: Path) -> Dict:
        """Validate PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            Validation result dictionary
        """
        errors = []

        # Check file exists
        if not file_path.exists():
            return {
                "valid": False,
                "errors": ["File does not exist"]
            }

        # Check file extension
        if file_path.suffix.lower() != '.pdf':
            errors.append("File must be a PDF (.pdf extension)")

        # Check file size
        file_size = file_path.stat().st_size
        if file_size > self.max_file_size:
            size_mb = file_size / (1024 * 1024)
            max_mb = self.max_file_size / (1024 * 1024)
            errors.append(f"File size ({size_mb:.1f}MB) exceeds maximum ({max_mb:.0f}MB)")

        # Check if file is readable
        try:
            with open(file_path, 'rb') as f:
                # Try to read first few bytes
                header = f.read(8)
                if not header.startswith(b'%PDF'):
                    errors.append("File does not appear to be a valid PDF")
        except Exception as e:
            errors.append(f"Cannot read file: {str(e)}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "file_size": file_size,
            "file_size_mb": file_size / (1024 * 1024)
        }

    def validate_uploaded_file(self, uploaded_file) -> Dict:
        """Validate uploaded file from Streamlit.

        Args:
            uploaded_file: Streamlit UploadedFile object

        Returns:
            Validation result dictionary
        """
        errors = []

        if uploaded_file is None:
            return {
                "valid": False,
                "errors": ["No file uploaded"]
            }

        # Check file extension
        if not uploaded_file.name.lower().endswith('.pdf'):
            errors.append("File must be a PDF (.pdf extension)")

        # Check file size
        file_size = uploaded_file.size
        if file_size > self.max_file_size:
            size_mb = file_size / (1024 * 1024)
            max_mb = self.max_file_size / (1024 * 1024)
            errors.append(f"File size ({size_mb:.1f}MB) exceeds maximum ({max_mb:.0f}MB)")

        # Check PDF header
        try:
            uploaded_file.seek(0)
            header = uploaded_file.read(8)
            uploaded_file.seek(0)

            if not header.startswith(b'%PDF'):
                errors.append("File does not appear to be a valid PDF")
        except Exception as e:
            errors.append(f"Cannot read file: {str(e)}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "file_size": file_size,
            "file_size_mb": file_size / (1024 * 1024),
            "file_name": uploaded_file.name
        }


class QueryValidator:
    """Validates user queries."""

    def __init__(self, min_length: int = 3, max_length: int = 500):
        """Initialize query validator.

        Args:
            min_length: Minimum query length
            max_length: Maximum query length
        """
        self.min_length = min_length
        self.max_length = max_length

    def validate_query(self, query: str) -> Dict:
        """Validate user query.

        Args:
            query: User's question

        Returns:
            Validation result dictionary
        """
        errors = []

        # Check if query is empty
        if not query or not query.strip():
            errors.append("Query cannot be empty")
            return {"valid": False, "errors": errors}

        query = query.strip()

        # Check length
        if len(query) < self.min_length:
            errors.append(f"Query too short (minimum {self.min_length} characters)")

        if len(query) > self.max_length:
            errors.append(f"Query too long (maximum {self.max_length} characters)")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "query": query,
            "length": len(query)
        }


def validate_detail_level(level: str) -> bool:
    """Validate detail level selection.

    Args:
        level: Detail level string

    Returns:
        True if valid
    """
    valid_levels = ["ejecutivo", "normal", "detallado"]
    return level.lower() in valid_levels


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for storage.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove or replace problematic characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')

    # Limit length
    name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
    if len(name) > 200:
        name = name[:200]

    return f"{name}.{ext}" if ext else name
