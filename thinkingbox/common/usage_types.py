# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional

from pydantic import BaseModel, Field


class InputTokensDetails(BaseModel):
    cached_tokens: int = 0


class OutputTokensDetails(BaseModel):
    reasoning_tokens: int = 0


class Usage(BaseModel):
    input_tokens: int = 0
    input_tokens_details: InputTokensDetails = Field(default_factory=InputTokensDetails)
    output_tokens: int = 0
    output_tokens_details: OutputTokensDetails = Field(
        default_factory=OutputTokensDetails
    )
    total_tokens: Optional[int] = None
