"""Storage service for local and cloud storage."""
import logging
from pathlib import Path
from typing import Optional, BinaryIO
from google.cloud import storage
from config import StorageType

logger = logging.getLogger(__name__)


class StorageService:
    """Handles file storage for local and cloud environments."""

    def __init__(self,
                 storage_type: StorageType,
                 local_path: Optional[str] = None,
                 gcs_bucket_name: Optional[str] = None):
        """Initialize storage service.

        Args:
            storage_type: Type of storage (local or cloud)
            local_path: Path for local storage
            gcs_bucket_name: GCS bucket name for cloud storage
        """
        self.storage_type = storage_type
        self.local_path = Path(local_path) if local_path else Path("./data")
        self.gcs_bucket_name = gcs_bucket_name

        self.gcs_client = None
        self.bucket = None

        if storage_type == StorageType.CLOUD:
            self._initialize_gcs()

    def _initialize_gcs(self):
        """Initialize Google Cloud Storage client."""
        try:
            logger.info("Initializing GCS client...")
            self.gcs_client = storage.Client()
            self.bucket = self.gcs_client.bucket(self.gcs_bucket_name)
            logger.info(f"GCS client initialized for bucket: {self.gcs_bucket_name}")

        except Exception as e:
            logger.error(f"Error initializing GCS: {e}")
            raise

    def save_file(self, file_data: BinaryIO, file_name: str, subdir: str = "") -> str:
        """Save file to storage.

        Args:
            file_data: File data (binary)
            file_name: Name of the file
            subdir: Optional subdirectory

        Returns:
            Path or URI to saved file
        """
        if self.storage_type == StorageType.LOCAL:
            return self._save_local(file_data, file_name, subdir)
        else:
            return self._save_gcs(file_data, file_name, subdir)

    def _save_local(self, file_data: BinaryIO, file_name: str, subdir: str) -> str:
        """Save file locally.

        Args:
            file_data: File data
            file_name: File name
            subdir: Subdirectory

        Returns:
            Local file path
        """
        try:
            # Create directory if needed
            save_dir = self.local_path / subdir
            save_dir.mkdir(parents=True, exist_ok=True)

            # Save file
            file_path = save_dir / file_name
            with open(file_path, 'wb') as f:
                f.write(file_data.read())

            logger.info(f"File saved locally: {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Error saving file locally: {e}")
            raise

    def _save_gcs(self, file_data: BinaryIO, file_name: str, subdir: str) -> str:
        """Save file to GCS.

        Args:
            file_data: File data
            file_name: File name
            subdir: Subdirectory (prefix)

        Returns:
            GCS URI
        """
        try:
            # Build blob name
            blob_name = f"{subdir}/{file_name}" if subdir else file_name

            # Upload to GCS
            blob = self.bucket.blob(blob_name)
            blob.upload_from_file(file_data, rewind=True)

            uri = f"gs://{self.gcs_bucket_name}/{blob_name}"
            logger.info(f"File saved to GCS: {uri}")
            return uri

        except Exception as e:
            logger.error(f"Error saving file to GCS: {e}")
            raise

    def load_file(self, file_path: str) -> bytes:
        """Load file from storage.

        Args:
            file_path: Path or URI to file

        Returns:
            File content as bytes
        """
        if self.storage_type == StorageType.LOCAL:
            return self._load_local(file_path)
        else:
            return self._load_gcs(file_path)

    def _load_local(self, file_path: str) -> bytes:
        """Load file from local storage.

        Args:
            file_path: Local file path

        Returns:
            File content
        """
        try:
            with open(file_path, 'rb') as f:
                return f.read()

        except Exception as e:
            logger.error(f"Error loading local file: {e}")
            raise

    def _load_gcs(self, file_uri: str) -> bytes:
        """Load file from GCS.

        Args:
            file_uri: GCS URI (gs://bucket/path)

        Returns:
            File content
        """
        try:
            # Parse URI
            if file_uri.startswith('gs://'):
                parts = file_uri[5:].split('/', 1)
                bucket_name = parts[0]
                blob_name = parts[1] if len(parts) > 1 else ""
            else:
                blob_name = file_uri

            # Download from GCS
            blob = self.bucket.blob(blob_name)
            return blob.download_as_bytes()

        except Exception as e:
            logger.error(f"Error loading file from GCS: {e}")
            raise

    def delete_file(self, file_path: str) -> bool:
        """Delete file from storage.

        Args:
            file_path: Path or URI to file

        Returns:
            Success status
        """
        if self.storage_type == StorageType.LOCAL:
            return self._delete_local(file_path)
        else:
            return self._delete_gcs(file_path)

    def _delete_local(self, file_path: str) -> bool:
        """Delete local file.

        Args:
            file_path: Local file path

        Returns:
            Success status
        """
        try:
            Path(file_path).unlink()
            logger.info(f"Deleted local file: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error deleting local file: {e}")
            return False

    def _delete_gcs(self, file_uri: str) -> bool:
        """Delete file from GCS.

        Args:
            file_uri: GCS URI

        Returns:
            Success status
        """
        try:
            # Parse URI
            if file_uri.startswith('gs://'):
                parts = file_uri[5:].split('/', 1)
                blob_name = parts[1] if len(parts) > 1 else ""
            else:
                blob_name = file_uri

            # Delete from GCS
            blob = self.bucket.blob(blob_name)
            blob.delete()

            logger.info(f"Deleted GCS file: {file_uri}")
            return True

        except Exception as e:
            logger.error(f"Error deleting GCS file: {e}")
            return False

    def list_files(self, prefix: str = "") -> list:
        """List files in storage.

        Args:
            prefix: Optional prefix to filter files

        Returns:
            List of file paths/URIs
        """
        if self.storage_type == StorageType.LOCAL:
            return self._list_local(prefix)
        else:
            return self._list_gcs(prefix)

    def _list_local(self, prefix: str) -> list:
        """List local files.

        Args:
            prefix: Subdirectory prefix

        Returns:
            List of file paths
        """
        search_path = self.local_path / prefix if prefix else self.local_path
        return [str(p) for p in search_path.rglob("*") if p.is_file()]

    def _list_gcs(self, prefix: str) -> list:
        """List GCS files.

        Args:
            prefix: Blob prefix

        Returns:
            List of GCS URIs
        """
        blobs = self.bucket.list_blobs(prefix=prefix)
        return [f"gs://{self.gcs_bucket_name}/{blob.name}" for blob in blobs]
