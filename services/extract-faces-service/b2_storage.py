"""
Backblaze B2 Object Storage Client
Handles uploading and deleting files from B2 using S3-compatible API
"""
import os
import io
from typing import Optional
import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
import structlog

logger = structlog.get_logger()


class B2StorageClient:
    """Client for interacting with Backblaze B2 object storage"""

    def __init__(
        self,
        key_id: str,
        app_key: str,
        bucket_name: str,
        region: str,
        endpoint: str
    ):
        """
        Initialize B2 storage client

        Args:
            key_id: B2 application key ID
            app_key: B2 application key
            bucket_name: Name of the B2 bucket
            region: B2 region (e.g., 'eu-central-003')
            endpoint: B2 S3-compatible endpoint URL
        """
        self.bucket_name = bucket_name
        self.endpoint = endpoint

        # Configure boto3 client for B2
        self.s3_client = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=key_id,
            aws_secret_access_key=app_key,
            region_name=region,
            config=BotoConfig(
                signature_version='s3v4',
                s3={'addressing_style': 'path'}
            )
        )

        logger.info(
            "B2 storage client initialized",
            bucket=bucket_name,
            region=region
        )

    def upload_headshot(
        self,
        image_data: bytes,
        person_id: str,
        user_id: str = "system"
    ) -> str:
        """
        Upload a face headshot image to B2

        Args:
            image_data: PNG image data as bytes
            person_id: ID of the person (used as filename)
            user_id: User ID for organizing files (default: "system")

        Returns:
            Public URL of the uploaded file

        Raises:
            Exception: If upload fails
        """
        # Construct file key (path in bucket)
        file_key = f"headshots/{user_id}/{person_id}.png"

        try:
            # Upload to B2
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=file_key,
                Body=image_data,
                ContentType="image/png",
                CacheControl="public, max-age=31536000",  # 1 year cache
                Metadata={
                    'person_id': person_id,
                    'user_id': user_id,
                    'uploaded_at': '',  # Will be set automatically
                }
            )

            # Construct public URL
            file_url = f"{self.endpoint}/{self.bucket_name}/{file_key}"

            logger.info(
                "Headshot uploaded to B2",
                person_id=person_id,
                file_key=file_key,
                url=file_url
            )

            return file_url

        except ClientError as e:
            logger.error(
                "Failed to upload headshot to B2",
                person_id=person_id,
                error=str(e)
            )
            raise Exception(f"Failed to upload headshot to B2: {str(e)}")

    def delete_headshot(
        self,
        person_id: str,
        user_id: str = "system"
    ) -> bool:
        """
        Delete a headshot from B2

        Args:
            person_id: ID of the person
            user_id: User ID (default: "system")

        Returns:
            True if successful, False otherwise
        """
        file_key = f"headshots/{user_id}/{person_id}.png"

        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=file_key
            )

            logger.info(
                "Headshot deleted from B2",
                person_id=person_id,
                file_key=file_key
            )

            return True

        except ClientError as e:
            logger.error(
                "Failed to delete headshot from B2",
                person_id=person_id,
                error=str(e)
            )
            # Don't raise - just log and return False
            return False


def get_b2_client() -> Optional[B2StorageClient]:
    """
    Factory function to create B2 storage client from environment variables

    Returns:
        B2StorageClient instance or None if credentials not configured
    """
    key_id = os.getenv("B2_KEY_ID")
    app_key = os.getenv("B2_APP_KEY")
    bucket_name = os.getenv("B2_BUCKET")
    region = os.getenv("B2_REGION")
    endpoint = os.getenv("B2_ENDPOINT")

    if not all([key_id, app_key, bucket_name, region, endpoint]):
        logger.warning(
            "B2 credentials not fully configured - headshot upload disabled",
            configured={
                "key_id": bool(key_id),
                "app_key": bool(app_key),
                "bucket_name": bool(bucket_name),
                "region": bool(region),
                "endpoint": bool(endpoint)
            }
        )
        return None

    return B2StorageClient(
        key_id=key_id,
        app_key=app_key,
        bucket_name=bucket_name,
        region=region,
        endpoint=endpoint
    )
