from pydantic import BaseModel, Field
from typing import List, Optional

class BiologyRequest(BaseModel):
    sequence: str = Field(..., description="Protein sequence")
    model_version: str = Field(default="2", pattern="^[12]$")
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    explain: bool = Field(default=False, description="Enable XAI explanation")
    mode: Optional[str] = Field(default=None, description="Inference mode, e.g., 'speculative'")

class MaterialsRequest(BaseModel):
    structure: str = Field(..., description="Material structure")
    model_version: str = Field(default="2", pattern="^[12]$")
    energy_threshold: float = Field(default=0.5, ge=0.0)
    explain: bool = Field(default=False, description="Enable XAI explanation")
    mode: Optional[str] = Field(default=None, description="Inference mode, e.g., 'speculative'")

class BatchBiologyRequest(BaseModel):
    requests: List[BiologyRequest]

class BatchMaterialsRequest(BaseModel):
    requests: List[MaterialsRequest]

class PipelineRequest(BaseModel):
    query: str = Field(..., description="The natural language query to initiate the pipeline.")

class LLMRequest(BaseModel):
    prompt: str = Field(..., description="Prompt for the LLM.")
    max_tokens: int = Field(default=100, ge=10, le=512)

class BenchmarkRequest(BaseModel):
    model_type: str = Field(..., description="Model to benchmark", pattern="^(bio|materials|llm)$")
    model_version: str = Field(default="1", description="Model version to benchmark")
    test_duration_seconds: int = Field(default=10, ge=5, le=60, description="Duration of the benchmark test in seconds.")

class DatasetRequest(BaseModel):
    model_type: str = Field(..., pattern="^(bio|materials)$")
    model_version: str = Field(default="2", pattern="^[12]$")
    size: int = Field(default=100, ge=10, le=1000)