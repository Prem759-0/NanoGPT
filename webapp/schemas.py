from typing import Optional
from pydantic import BaseModel, Field, field_validator

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Prompt or input message for the language model"
    )
    max_new_tokens: int = Field(
        default=150,
        ge=10,
        le=500,
        description="Maximum number of tokens to generate"
    )
    temperature: float = Field(
        default=0.8,
        ge=0.1,
        le=2.0,
        description="Sampling temperature (lower = more conservative, higher = more creative)"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for conversation memory"
    )

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Message cannot be empty or whitespace only.")
        return trimmed

class ChatResponse(BaseModel):
    reply: str
    model_name: str
    prompt_length: int
    tokens_generated: int
    device: str
    session_id: str

class SessionItem(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str

class CreateSessionRequest(BaseModel):
    title: Optional[str] = "New Chat"


class HealthResponse(BaseModel):
    status: str
    model_status: str
    checkpoint_exists: bool
    model_loaded: bool
    device: str
    torch_available: bool
    torch_version: Optional[str] = None

class ModelInfoResponse(BaseModel):
    model_type: str
    checkpoint_path: Optional[str] = None
    model_status: str
    dataset: Optional[str] = None
    vocab_size: Optional[int] = None
    parameters: Optional[int] = None
    block_size: Optional[int] = None
    n_layer: Optional[int] = None
    n_head: Optional[int] = None
    n_embd: Optional[int] = None
    description: str

class ImageGenRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500, description="Image generation prompt")

class ImageGenResponse(BaseModel):
    image_url: str
    prompt: str
    width: int
    height: int

class CodeRunRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=5000, description="Python code snippet to execute")

class CodeRunResponse(BaseModel):
    success: bool
    output: str
    error: Optional[str] = None
    exit_code: int
    execution_time_ms: float

class SwitchModelRequest(BaseModel):
    model_name: str = Field(..., description="Target model name: 'gpt2' or 'scratch'")


