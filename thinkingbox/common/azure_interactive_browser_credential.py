# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os

from azure.identity import (
    AuthenticationRecord,
    InteractiveBrowserCredential,
    TokenCachePersistenceOptions,
)

from thinkingbox.common.utils import CredentialBase


class AzureInteractiveBrowserCredential(CredentialBase):
    r"""
    Allows obtaining Azure Entra ID tokens by opening a browser window for user login. Before opening the browser window the class will show a descriptive method in the console about the purpose of the authentication (e.g. Azure AI Foundry access or MCP server access). Once authenticated the identity is cached for future use and tokens get refreshed automatically. The cache is stored in the user's local application data folder (e.g. c:\users\<username>\appdata\local\ThinkingBox\AuthRecords for Windows).
    """

    _logger = logging.getLogger(__name__)

    @classmethod
    def force_reauthentication(cls):
        """
        Deletes all authentication record files in the AuthRecords directory.
        Use this to force reauthentication for all cached credentials.
        """
        auth_records_path = AzureInteractiveBrowserCredential._get_auth_records_path()
        if os.path.exists(auth_records_path):
            for filename in os.listdir(auth_records_path):
                file_path = os.path.join(auth_records_path, filename)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                        cls._logger.debug(f"Deleted auth record file: {file_path}")
                    except Exception as e:
                        cls._logger.error(
                            f"Failed to delete auth record file {file_path}: {e}"
                        )
                        raise (e)

        cls._logger.info(
            "All authentication record files deleted. You will be prompted to reauthenticate on next use."
        )

    def __init__(
        self,
        name,
        scope,
        purpose: str,
        account_hint: str = None,
        client_id=None,
        tenant_id=None,
    ):
        self._scope = scope
        self.purpose = purpose
        self.account_hint = account_hint
        auth_records_path = AzureInteractiveBrowserCredential._get_auth_records_path()
        self._record_path = os.path.join(auth_records_path, name + ".json")
        self._client_id = client_id
        self._tenant_id = tenant_id
        self._credential = None
        self._ensure_authenticated()
        self._logger.debug(
            f"AzureInteractiveBrowserCredential initialized for endpoint: {name}"
        )

    async def get_token(self) -> str:
        """
        Retrieves an access token for the configured scope.
        If authentication fails, attempts reauthentication and retries.
        """
        try:
            token = self._credential.get_token(self._scope)
        except Exception:
            self._logger.warning(
                f"\033[92m\033[43mFailed to retrieve token for: {self.purpose}\033[0m"
            )

            if os.path.exists(self._record_path):
                try:
                    os.remove(self._record_path)
                    self._logger.info(
                        f"Deleted cached authentication record: {self._record_path}"
                    )
                except Exception as del_err:
                    self._logger.error(f"Failed to delete auth record: {del_err}")
                    raise (del_err)

            self._ensure_authenticated()
            token = self._credential.get_token(self._scope)

        self._logger.debug(f"Token successfully retrieved for: {self.purpose}")
        return token.token

    @staticmethod
    def _get_auth_records_path():
        result = AzureInteractiveBrowserCredential._get_local_app_data()
        result = os.path.join(result, "ThinkingBox")
        result = os.path.join(result, "AuthRecords")
        os.makedirs(result, exist_ok=True)

        return result

    @staticmethod
    def _get_local_app_data():
        import os
        import platform

        system = platform.system()
        if system == "Windows":
            return os.getenv("LOCALAPPDATA")
        elif system == "Linux":
            return os.path.expanduser("~/.local/share")
        else:
            raise RuntimeError(f"Unsupported platform: {system}")

    def _ensure_authenticated(self):
        # Try to load authentication record
        auth_record = None
        if os.path.exists(self._record_path):
            try:
                with open(self._record_path, "r") as f:
                    serialized_record = f.read()
                auth_record = AuthenticationRecord.deserialize(serialized_record)
            except Exception:
                auth_record = None

        credential_args = {"cache_persistence_options": TokenCachePersistenceOptions()}
        if self._client_id is not None:
            credential_args["client_id"] = self._client_id
        if self._tenant_id is not None:
            credential_args["tenant_id"] = self._tenant_id

        if auth_record is None:
            print(
                f"\033[92m\033[43mYou are about to authenticate for: {self.purpose}. A browser window will open for authentication.\033[0m"
            )

            if self.account_hint is not None:
                print(f"\033[92m\033[43mAccount hint: {self.account_hint}\033[0m")

            print(
                "\033[92m\033[43mIf, at a later point, you need to re-authenticate, run tb infer with the -reauthenticate flag.\033[0m"
            )
            proceed = input("Proceed with authentication? (y/n): ").strip().lower()
            if proceed != "y":
                raise RuntimeError("Authentication cancelled by user.")
            self._credential = InteractiveBrowserCredential(**credential_args)
            auth_record = self._credential.authenticate(scopes=[self._scope])
            with open(self._record_path, "w") as f:
                f.write(auth_record.serialize())

            print("Authentication successful.")

            self._logger.info(
                f"Authentication successful. Saved authentication record to: {self._record_path}"
            )
        else:
            self._credential = InteractiveBrowserCredential(
                authentication_record=auth_record, **credential_args
            )

    async def invalidate(self):
        # The token is refreshed as necessary in get_token by azure.identity
        pass
