from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.elements import ElementBinding


AspectRatio = Literal["16:9", "9:16", "1:1"]
RenderQuality = Literal["preview", "1080p", "4k"]
AudioMode = Literal["native", "mastered", "mute"]
VideoProviderName = Literal["huggingface", "modal"]
VideoModelName = Literal["ltx-2.5", "wan", "minimax"]
DecoderName = Literal["conv", "diffusion"]
ContinuityMode = Literal["off", "balanced", "strict"]
ContinuityQCMode = Literal["off", "auto", "strict"]
QualityCheckStatus = Literal["passed", "failed", "unavailable", "not_checked"]
RealismProfile = Literal["standard", "real_skin", "identity_max"]
JobStatusName = Literal["queued", "running", "completed", "failed"]
JobStageName = Literal[
    "queued",
    "initializing",
    "planning",
    "rendering",
    "composing",
    "delivery",
    "probing",
    "billing",
    "publishing",
    "completed",
    "failed",
]


class PlanQualityReport(BaseModel):
    coverage_score: float
    covered_terms: list[str]
    missing_terms: list[str]
    note: str


class ScenePlanRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=50000)
    aspect_ratio: AspectRatio = "16:9"
    scene_count: int = Field(default=4, ge=1, le=20)


class EntityLock(BaseModel):
    label: str = Field(..., min_length=1, max_length=64)
    expected_count: int = Field(default=1, ge=1, le=8)
    description: str = Field(default="", max_length=1200)


class Scene(BaseModel):
    id: int
    title: str
    prompt: str
    duration_seconds: int
    # Exact subject counts for this shot when the planner knows them. Empty means
    # "use the global maximum locks but do not force every entity to be visible".
    visible_entity_counts: dict[str, int] = Field(default_factory=dict)


class ScenePlanResponse(BaseModel):
    original_prompt: str
    aspect_ratio: AspectRatio
    scenes: list[Scene]
    plan_quality: PlanQualityReport | None = None
    planner_source: Literal["gemini", "direct", "fallback"] = "gemini"
    planner_note: str | None = None
    continuity_id: str
    character_bible: str
    style_bible: str
    entity_locks: list[EntityLock] = Field(default_factory=list)


class VideoGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=10, max_length=8000)
    aspect_ratio: AspectRatio = "16:9"
    duration_seconds: float = Field(default=5.0, ge=1.0, le=30.0)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    decoder: DecoderName = "conv"
    enhance_prompt: bool = False
    quality: RenderQuality = "preview"
    realism_profile: RealismProfile = "standard"
    audio_mode: AudioMode = "native"
    audio_direction: str | None = Field(default=None, max_length=1200)
    provider: VideoProviderName | None = None
    model: VideoModelName = "ltx-2.5"

    # Multi-scene continuity. `strict` adds LTX first-frame conditioning when a
    # previous scene frame is supplied; `balanced` keeps the text identity/style
    # locks and seed without image conditioning.
    continuity_mode: ContinuityMode = "off"
    continuity_id: str | None = Field(default=None, max_length=96)
    scene_index: int | None = Field(default=None, ge=0, le=49)
    scene_count: int | None = Field(default=None, ge=1, le=50)
    character_bible: str | None = Field(default=None, max_length=3000)
    style_bible: str | None = Field(default=None, max_length=3000)
    entity_locks: list[EntityLock] = Field(default_factory=list, max_length=12)
    visible_entity_counts: dict[str, int] = Field(default_factory=dict)
    continuity_qc_mode: ContinuityQCMode = "auto"
    continuity_max_retries: int = Field(default=1, ge=0, le=2)
    reference_frame_filename: str | None = Field(default=None, max_length=255)
    continuity_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    element_bindings: list[ElementBinding] = Field(default_factory=list, max_length=20)


class MediaInfo(BaseModel):
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    video_codec: str | None = None
    has_audio: bool = False
    audio_codec: str | None = None
    audio_channels: int | None = None
    format_name: str | None = None
    size_bytes: int = 0


class VideoGenerationResponse(BaseModel):
    video_url: str
    download_url: str
    filename: str
    seed: int
    render_details: str
    render_seconds: float
    wall_seconds: float
    provider: str
    model: str
    quality: RenderQuality
    quality_note: str
    realism_profile: RealismProfile = "standard"
    detail_refined: bool = False
    audio_mode: AudioMode = "native"
    chunk_count: int = 1
    gpu: str | None = None
    media_info: MediaInfo
    estimated_cost_usd: float | None = None
    estimated_cost_per_output_minute_usd: float | None = None
    cost_note: str
    continuity_mode: ContinuityMode = "off"
    continuity_applied: bool = False
    reference_frame_filename: str | None = None
    continuity_frame_url: str | None = None
    continuity_frame_filename: str | None = None
    continuity_qc_passed: bool | None = None
    visual_qc_status: QualityCheckStatus = "not_checked"
    continuity_regenerations: int = 0
    continuity_warnings: list[str] = Field(default_factory=list)
    elements_used: list[str] = Field(default_factory=list)
    element_reference_mode: str | None = None


class FullVideoScene(BaseModel):
    id: int
    prompt: str = Field(..., min_length=10, max_length=8000)
    visible_entity_counts: dict[str, int] = Field(default_factory=dict)


class FullVideoGenerationRequest(BaseModel):
    scenes: list[FullVideoScene] = Field(..., min_length=1, max_length=20)
    aspect_ratio: AspectRatio = "16:9"
    duration_seconds: float = Field(default=5.0, ge=1.0, le=30.0)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    decoder: DecoderName = "conv"
    enhance_prompt: bool = False
    quality: RenderQuality = "preview"
    realism_profile: RealismProfile = "standard"
    audio_mode: AudioMode = "native"
    audio_direction: str | None = Field(default=None, max_length=1200)
    provider: VideoProviderName | None = None
    model: VideoModelName = "ltx-2.5"
    continuity_mode: ContinuityMode = "strict"
    continuity_id: str | None = Field(default=None, max_length=96)
    character_bible: str | None = Field(default=None, max_length=3000)
    style_bible: str | None = Field(default=None, max_length=3000)
    entity_locks: list[EntityLock] = Field(default_factory=list, max_length=12)
    continuity_qc_mode: ContinuityQCMode = "auto"
    continuity_max_retries: int = Field(default=1, ge=0, le=2)
    continuity_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    element_bindings: list[ElementBinding] = Field(default_factory=list, max_length=20)


class FullVideoGenerationResponse(BaseModel):
    final_video_url: str
    final_download_url: str
    final_filename: str
    scene_video_urls: list[str]
    render_details: list[str]
    total_render_seconds: float
    total_wall_seconds: float
    provider: str
    model: str
    quality: RenderQuality
    quality_note: str
    realism_profile: RealismProfile = "standard"
    detail_refined: bool = False
    audio_mode: AudioMode = "native"
    gpu: str | None = None
    media_info: MediaInfo
    estimated_cost_usd: float | None = None
    estimated_cost_per_output_minute_usd: float | None = None
    cost_note: str


class CombineScenesRequest(BaseModel):
    scene_video_urls: list[str] = Field(..., min_length=1, max_length=50)
    aspect_ratio: AspectRatio = "16:9"
    quality: RenderQuality = "preview"
    audio_mode: AudioMode = "native"


class CombineScenesResponse(BaseModel):
    final_video_url: str
    final_download_url: str
    final_filename: str
    scene_count: int
    quality: RenderQuality
    quality_note: str
    audio_mode: AudioMode = "native"
    media_info: MediaInfo


class GenerationCapabilitiesResponse(BaseModel):
    providers: list[dict]
    models: list[dict]
    qualities: list[dict]
    aspect_ratios: list[AspectRatio]
    decoders: list[DecoderName]
    realism_profiles: list[dict]
    continuity_modes: list[ContinuityMode]
    audio_modes: list[AudioMode]
    image_conditioning: bool
    entity_count_lock: bool = True
    continuity_vision_qc: bool = True
    max_scene_duration_seconds: float
    max_scene_duration_seconds_by_quality: dict[str, float]
    native_chunk_seconds: float
    max_factory_duration_seconds: int
    async_jobs: bool
    audio_probe: bool
    cost_tracking_configured: bool
    gpu: str | None = None
    production_mode: bool = False
    job_workers: int = 1
    job_max_pending: int = 3
    elements: dict | None = None


class AsyncVideoGenerationResponse(BaseModel):
    job_id: str
    status: JobStatusName
    status_url: str


class GenerationJobResponse(BaseModel):
    job_id: str
    job_type: str
    status: JobStatusName
    stage: JobStageName
    progress: int
    message: str
    payload: dict
    result: dict | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class MetricsSummaryResponse(BaseModel):
    total_events: int
    total_render_seconds: float
    total_estimated_cost_usd: float
    average_render_seconds: float | None = None
    average_estimated_cost_usd: float | None = None
    by_gpu: dict[str, dict]
