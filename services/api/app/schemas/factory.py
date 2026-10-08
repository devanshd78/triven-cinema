from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.generation import (
    AspectRatio,
    AudioMode,
    ContinuityMode,
    ContinuityQCMode,
    EntityLock,
    DecoderName,
    RenderQuality,
    QualityCheckStatus,
    RealismProfile,
    VideoModelName,
    VideoProviderName,
)
from app.schemas.elements import ElementBinding
from app.schemas.youtube import YouTubePrivacy


class FactoryGenerationRequest(BaseModel):
    spoken_script: str = Field(default="", max_length=50000)
    request_id: str | None = Field(default=None, max_length=128)
    chat_id: str | None = Field(default=None, max_length=128)
    prompt: str = Field(..., min_length=10, max_length=50000)
    target_duration_seconds: float = Field(default=30.0, ge=15.0, le=300.0)
    scene_duration_seconds: float = Field(default=20.0, ge=15.0, le=30.0)
    aspect_ratio: AspectRatio = "16:9"
    quality: RenderQuality = "1080p"
    realism_profile: RealismProfile = "real_skin"
    audio_mode: AudioMode = "mastered"
    audio_direction: str | None = Field(default=None, max_length=1200)
    provider: VideoProviderName = "modal"
    model: VideoModelName = "ltx-2.5"
    decoder: DecoderName = "conv"
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    continuity_mode: ContinuityMode = "strict"
    continuity_strength: float = Field(default=0.95, ge=0.0, le=1.0)
    continuity_qc_mode: ContinuityQCMode = "strict"
    continuity_max_retries: int = Field(default=1, ge=0, le=2)
    enhance_prompt: bool = False

    # Reusable workspace Elements referenced with @handles in the prompt. The job
    # stores immutable element/version ids rather than relying on plain text names.
    element_bindings: list[ElementBinding] = Field(default_factory=list, max_length=20)

    publish_to_youtube: bool = False
    youtube_title: str | None = Field(default=None, max_length=100)
    youtube_description: str = Field(default="", max_length=5000)
    youtube_privacy: YouTubePrivacy = "private"
    youtube_tags: list[str] = Field(default_factory=list, max_length=30)
    youtube_category_id: str = Field(default="22", max_length=8)
    youtube_publish_at: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def validate_factory_request(self):
        if self.scene_duration_seconds > self.target_duration_seconds:
            self.scene_duration_seconds = self.target_duration_seconds
        if self.publish_to_youtube and not (self.youtube_title or "").strip():
            # A safe title is also derived at runtime, so this is deliberately not an error.
            self.youtube_title = None
        return self


class FactoryGenerationResponse(BaseModel):
    final_video_url: str
    final_download_url: str
    final_filename: str
    target_duration_seconds: float
    actual_duration_seconds: float | None = None
    scene_count: int
    scene_duration_seconds: float
    aspect_ratio: AspectRatio
    quality: RenderQuality
    quality_note: str
    realism_profile: RealismProfile = "real_skin"
    detail_refined: bool = False
    audio_mode: AudioMode
    has_audio: bool
    width: int | None = None
    height: int | None = None
    provider: str
    model: str
    gpu: str | None = None
    total_render_seconds: float
    total_wall_seconds: float
    chunk_count: int
    estimated_cost_usd: float | None = None
    estimated_cost_per_output_minute_usd: float | None = None
    cost_note: str
    planner_source: str
    planner_note: str | None = None
    continuity_id: str
    entity_locks: list[EntityLock] = Field(default_factory=list)
    continuity_qc_passed: bool | None = None
    visual_qc_status: QualityCheckStatus = "not_checked"
    continuity_regenerations: int = 0
    continuity_warnings: list[str] = Field(default_factory=list)
    audio_qc_passed: bool | None = None
    audio_qc_status: QualityCheckStatus = "not_checked"
    audio_retake_count: int = 0
    audio_warnings: list[str] = Field(default_factory=list)
    delivery_complete: bool = True
    scene_results: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    elements_used: list[str] = Field(default_factory=list)
    element_reference_mode: str | None = None
    youtube_video_id: str | None = None
    youtube_url: str | None = None
    youtube_privacy: Literal["private", "unlisted", "public"] | None = None
