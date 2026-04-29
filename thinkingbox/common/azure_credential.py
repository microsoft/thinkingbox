# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import json
import os
import threading
import time
from typing import Literal

import httpx

from thinkingbox.common.utils import CredentialBase


class AsyncThreadLock:
    def __init__(self):
        self._lock = threading.Lock()

    async def __aenter__(self):
        await self._acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._release()

    async def _acquire(self):
        await asyncio.get_running_loop().run_in_executor(None, self._lock.acquire)

    def _release(self):
        self._lock.release()


class AzureCredential(CredentialBase):
    def __init__(
        self,
        resource="https://cognitiveservices.azure.com/",
        type: Literal["az-cli", "managed-identity"] = "az-cli",
        duration: float = 600.0,
        client_id: str | None = None,
    ):
        self._token = None
        self._resource = resource
        self._type = type
        self._duration = duration
        self._token_expiry_time = 0
        self._client_id = client_id
        if self._client_id is None and self._type == "managed-identity":
            self._client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
        if self._type not in ("az-cli", "managed-identity"):
            raise ValueError(f"Invalid credential type for Azure OpenAI: {self._type}")
        self._lock = AsyncThreadLock()

    async def _fetch_new_token(self) -> str:
        if self._type == "az-cli":
            return await self._fetch_new_token_az_cli()
        elif self._type == "managed-identity":
            return await self._fetch_new_token_mi()
        else:
            raise RuntimeError(f"Unknown credential type: {self._type}")

    async def _fetch_new_token_mi(self) -> str:
        try:
            msi_endpoint = os.environ["MSI_ENDPOINT"]
            msi_secret = os.environ["MSI_SECRET"]
            params = {"resource": self._resource}
            if self._client_id:
                params["clientid"] = self._client_id
            headers = {"Secret": msi_secret}

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(msi_endpoint, params=params, headers=headers)
                resp.raise_for_status()
                token_data = resp.json()
                return token_data["access_token"]
        except Exception as e:
            raise RuntimeError(
                f"Failed to retrieve Azure OpenAI token (via managed identity): {str(e)}"
            )

    async def _fetch_new_token_az_cli(self) -> str:
        try:
            process = await asyncio.create_subprocess_exec(
                "az.cmd" if (os.name == "nt") else "az",
                "account",
                "get-access-token",
                "--resource",
                self._resource,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                raise RuntimeError(
                    f"Azure CLI command failed: {stderr.decode().strip()}"
                )

            token_data = json.loads(stdout.decode())
            return token_data["accessToken"]
        except Exception as e:
            raise RuntimeError(
                f"Failed to retrieve Azure OpenAI token (via az-cli): {str(e)}"
            )

    async def get_token(self) -> str:
        async with self._lock:
            token = self._token
            current_time = time.monotonic()
            if self._token is None or current_time >= self._token_expiry_time:
                self._token = await self._fetch_new_token()
                self._token_expiry_time = current_time + self._duration
                token = self._token
        return token

    async def invalidate(self) -> None:
        # next call to get_token will not use the cached token
        async with self._lock:
            self._token_expiry_time = 0.0


class AzureIdentityCredential(CredentialBase):
    def __init__(self, name: str, scope: str, *args, **kwargs):
        import azure.identity

        self.scope = scope
        self.az_credential = getattr(azure.identity, name)(*args, **kwargs)

    async def get_token(self):
        token = await asyncio.to_thread(self.az_credential.get_token, self.scope)
        return token.token

    async def invalidate(self):
        # This is done automatically in get_token
        pass
