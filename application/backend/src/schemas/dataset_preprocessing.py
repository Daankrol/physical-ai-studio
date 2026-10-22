from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PreprocessingFeature(BaseModel):
    key: str
    shape: list[int]
    is_depth: bool = False
    unit: str | None = None
    aligned_to: str | None = None


class DatasetPreprocessingInfo(BaseModel):
    features: list[PreprocessingFeature]
    frames: int
    has_teacher_cache: bool = False


class DatasetPreprocessingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal["depth", "teacher_targets"]
    rgb_key: str | None = None
    depth_key: str | None = None
    depth_aligned: bool = False
    chunk_size: int = Field(default=50, ge=1, le=1000)
    batch_size: int = Field(default=1, ge=1, le=64, strict=True, description="RGB frames per depth inference batch")
    device_index: int = Field(default=0, ge=0, le=63)

    @model_validator(mode="after")
    def validate_feature_names(self) -> "DatasetPreprocessingRequest":
        for key in (self.rgb_key, self.depth_key):
            if key is not None and (not key.startswith("observation.") or "/" in key or "\\" in key or ".." in key):
                raise ValueError("Select valid observation feature names")
        if self.operation == "depth" and (self.rgb_key is not None or self.depth_key is not None):
            raise ValueError("Depth computation automatically processes all RGB cameras")
        if self.operation == "teacher_targets" and self.rgb_key is None:
            raise ValueError("Teacher targets need an RGB source camera")
        if self.operation == "teacher_targets" and self.batch_size != 1:
            raise ValueError("Teacher-target caching supports batch size 1")
        return self


class DatasetPreprocessingPayload(DatasetPreprocessingRequest):
    dataset_id: UUID

    @model_validator(mode="before")
    @classmethod
    def migrate_derived_payload(cls, value: object) -> object:
        """Keep jobs stored before in-place preprocessing readable."""
        if isinstance(value, dict) and ("output_name" in value or "result_dataset_id" in value):
            value = {key: item for key, item in value.items() if key not in {"output_name", "result_dataset_id"}}
            if value.get("operation") == "depth":
                value.update(rgb_key=None, depth_key=None)
        return value
