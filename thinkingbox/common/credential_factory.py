# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path

from thinkingbox.common.azure_credential import (
    AzureCredential,
    AzureIdentityCredential,
)
from thinkingbox.common.config_types import (
    ApiKeyCredentialConfig,
    AzureCliCredentialConfig,
    AzureIdentityCredentialConfig,
    AzureManagedIdentityCredentialConfig,
    CredentialConfigT,
)
from thinkingbox.common.utils import ApiKeyCredential


def create_credential(config: CredentialConfigT):
    if isinstance(config, ApiKeyCredentialConfig):
        return ApiKeyCredential(config.api_key)
    if isinstance(config, AzureCliCredentialConfig):
        return AzureCredential(
            type="az-cli",
            resource=config.resource,
            duration=config.duration,
        )
    if isinstance(config, AzureManagedIdentityCredentialConfig):
        return AzureCredential(
            type="managed-identity",
            resource=config.resource,
            client_id=config.client_id,
            duration=config.duration,
        )
    if isinstance(config, AzureIdentityCredentialConfig):
        cred_kwargs = config.model_extra.copy()
        if "certificate_path" in cred_kwargs:
            cred_kwargs["certificate_path"] = Path(
                cred_kwargs["certificate_path"]
            ).expanduser()
        return AzureIdentityCredential(
            config.name,
            config.scope,
            **cred_kwargs,
        )
    raise TypeError(f"Invalid credential config type {type(config)}")
