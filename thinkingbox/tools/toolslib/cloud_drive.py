# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import datetime
import fnmatch
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field, field_serializer

DT_EPOCH = datetime.datetime(2024, 4, 1, 10, 10, 0, tzinfo=datetime.timezone.utc)


def parse_dt_always_utc(value):
    if isinstance(value, datetime.datetime):
        return value.astimezone(datetime.timezone.utc)
    return datetime.datetime.fromisoformat(value).replace(tzinfo=datetime.timezone.utc)


class Metadata(BaseModel):
    created: Annotated[datetime.datetime, BeforeValidator(parse_dt_always_utc)] = (
        DT_EPOCH
    )
    modified: Annotated[datetime.datetime, BeforeValidator(parse_dt_always_utc)] = (
        DT_EPOCH
    )

    @field_serializer("created", "modified")
    def serialize_dt(self, dt: datetime.datetime, _info):
        return dt.strftime("%Y-%m-%dT%H:%M:%S")


class File(BaseModel):
    path: str  # e.g. path/to/file.txt
    text_content: str
    metadata: Metadata = Field(default_factory=Metadata)


class SearchResult(BaseModel):
    path: str
    created: Annotated[datetime.datetime, BeforeValidator(parse_dt_always_utc)] = (
        DT_EPOCH
    )
    modified: Annotated[datetime.datetime, BeforeValidator(parse_dt_always_utc)] = (
        DT_EPOCH
    )

    @field_serializer("created", "modified")
    def serialize_dt(self, dt: datetime.datetime, _info):
        return dt.strftime("%Y-%m-%dT%H:%M:%S")


class CloudDriveError(Exception):
    pass


class CloudDrive:
    def __init__(self, files):
        """
        Initialize the CloudDrive instance.

        Args:
            files (list[File]): A list of File objects representing the initial files in the cloud drive.
        """
        self.files: list[File] = files
        self.effects = []
        self.time = DT_EPOCH + datetime.timedelta(hours=30)

    def get_time(self) -> datetime.datetime:
        return self.time

    def advance_time(self):
        self.time += datetime.timedelta(seconds=60)

    def get_metadata(self, path: str) -> Metadata:
        """
        Get metadata of a file.

        Args:
            path (str): The path of the file.

        Returns:
            Metadata: Metadata of the file.

        Raises:
            CloudDriveError: If the file does not exist.
        """
        for file in self.files:
            if file.path == path:
                return file.metadata
        raise CloudDriveError(f"File not found: {path}")

    def upload_file(self, path: str, text_content: str, overwrite: bool) -> None:
        """
        Upload a file to the cloud drive.

        Args:
            path (str): The path where the file will be uploaded.
            text_content (str): The content of the file.
            overwrite (bool): Whether to overwrite the file if it already exists.

        Raises:
            CloudDriveError: If the file exists and overwrite is False.
        """
        t = self.get_time()

        for file in self.files:
            if file.path == path:
                if not overwrite:
                    raise CloudDriveError(f"File already exists: {path}")
                file.text_content = text_content
                file.metadata.modified = t
                self.effects.append(
                    {
                        "type": "overwrite",
                        "path": file.path,
                        "text_content": file.text_content,
                    }
                )
                return

        new_file = File(
            path=path,
            metadata=Metadata(
                created=t,
                modified=t,
            ),
            text_content=text_content,
        )
        self.effects.append(
            {
                "type": "create",
                "path": new_file.path,
                "text_content": new_file.text_content,
            }
        )
        self.advance_time()
        self.files.append(new_file)

    def delete_file(self, path: str) -> None:
        """
        Delete a file from the cloud drive.

        Args:
            path (str): The path of the file to delete.

        Raises:
            CloudDriveError: If the file does not exist.
        """
        for file in self.files:
            if file.path == path:
                self.effects.append({"type": "delete", "path": file.path})
                self.advance_time()
                self.files.remove(file)
                return
        raise CloudDriveError(f"File not found: {path}")

    def get_text_content(self, path: str) -> str:
        """
        Get the text content of a file.

        Args:
            path (str): The path of the file.

        Returns:
            str: The text content of the file.

        Raises:
            CloudDriveError: If the file does not exist.
        """
        for file in self.files:
            if file.path == path:
                return file.text_content
        raise CloudDriveError(f"File not found: {path}")

    def copy_file(
        self, source_path: str, destination_path: str, overwrite: bool
    ) -> None:
        """
        Copy a file from source to destination.

        Args:
            source_path (str): The path of the source file.
            destination_path (str): The path of the destination file.
            overwrite (bool): Whether to overwrite the destination file if it exists.

        Raises:
            CloudDriveError: If the source file does not exist or the destination file exists and overwrite is False.
        """
        source_file = None
        for file in self.files:
            if file.path == source_path:
                source_file = file
                break

        if not source_file:
            raise CloudDriveError(f"Source file not found: {source_path}")

        t = self.get_time()
        for file in self.files:
            if file.path == destination_path:
                if not overwrite:
                    raise CloudDriveError(
                        f"Destination file already exists: {destination_path}"
                    )
                file.text_content = source_file.text_content
                file.metadata.modified = t
                self.effects.append(
                    {
                        "type": "overwrite",
                        "path": file.path,
                        "text_content": file.text_content,
                    }
                )
                self.advance_time()
                return

        new_file = File(
            path=destination_path,
            metadata=Metadata(
                created=t,
                modified=t,
            ),
            text_content=source_file.text_content,
        )
        self.effects.append(
            {
                "type": "create",
                "path": new_file.path,
                "text_content": new_file.text_content,
            }
        )
        self.advance_time()
        self.files.append(new_file)

    def list_files(
        self,
        prefix: str,
        sort: Literal["name", "created", "modified"] = "name",
        order: Literal["ascending", "descending"] = "ascending",
    ) -> list[SearchResult]:
        """
        List files whose paths start with a given prefix.

        Args:
            prefix (str): The prefix to filter files by.
            sort (Literal["name", "created", "modified"]): The attribute to sort files by. Defaults to "name".
            order (Literal["ascending", "descending"]): The sort order. Defaults to "ascending".

        Returns:
            list[SearchResult]: A list of files matching the prefix.
        """
        filtered_files = [
            SearchResult(
                path=file.path,
                created=file.metadata.created,
                modified=file.metadata.modified,
            )
            for file in self.files
            if file.path.startswith(prefix)
        ]

        reverse = order == "descending"
        if sort == "name":
            filtered_files.sort(key=lambda x: x.path, reverse=reverse)
        elif sort == "created":
            filtered_files.sort(key=lambda x: x.created, reverse=reverse)
        elif sort == "modified":
            filtered_files.sort(key=lambda x: x.modified, reverse=reverse)

        return filtered_files

    def find_files(
        self,
        pattern: str,
        sort: Literal["name", "created", "modified"] = "name",
        order: Literal["ascending", "descending"] = "ascending",
    ) -> list[SearchResult]:
        """
        Find files whose paths match a given pattern.

        Args:
            pattern (str): The pattern to match file paths against.
            sort (Literal["name", "created", "modified"]): The attribute to sort files by. Defaults to "name".
            order (Literal["ascending", "descending"]): The sort order. Defaults to "ascending".

        Returns:
            list[SearchResult]: A list of files matching the pattern.
        """
        filtered_files = [
            SearchResult(
                path=file.path,
                created=file.metadata.created,
                modified=file.metadata.modified,
            )
            for file in self.files
            if fnmatch.fnmatch(file.path, pattern)
        ]

        reverse = order == "descending"
        if sort == "name":
            filtered_files.sort(key=lambda x: x.path, reverse=reverse)
        elif sort == "created":
            filtered_files.sort(key=lambda x: x.created, reverse=reverse)
        elif sort == "modified":
            filtered_files.sort(key=lambda x: x.modified, reverse=reverse)

        return filtered_files
