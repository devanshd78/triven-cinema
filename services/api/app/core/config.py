from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[4]
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    app_name: str = "Triven Cinema API"
    app_env: str = "development"
    debug: bool = True

    # Browser access. In production the Next.js app proxies /api and /media to the
    # loopback-only API, so CORS can stay disabled/empty.
    frontend_url: str = "http://localhost:3000"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    gemini_timeout_seconds: float = 20.0
    gemini_thinking_level: str = "low"
    gemini_fallback_models: str = "gemini-3.7-flash,gemini-3.6-flash"
    gemini_max_attempts_per_model: int = 3
    gemini_retry_backoff_seconds: float = 0.8

    video_provider: str = "huggingface"
    hf_token: str = ""
    hf_ltx_space: str = "ChopperBlu/ltx-2-5-demo"

    modal_app_name: str = "triven-cinema-ltx"
    modal_function_name: str = "generate_video"
    triven_modal_gpu: str = "B200"
    triven_ltx_repo_ref: str = "v1.4.2"

    # Keep these at 0 until you copy the current hourly rates from Modal.
    # They are used only for explicit cost estimates, never presented as billed cost.
    modal_gpu_hourly_usd_b200: float = 0.0
    modal_gpu_hourly_usd_h200: float = 0.0
    modal_gpu_hourly_usd_h100: float = 0.0

    default_render_quality: str = "preview"
    default_decoder: str = "conv"
    default_scene_duration_seconds: float = 20.0

    # LTX-2.5's native duration head is designed around clips up to 20 seconds.
    # Factory keeps 15-20s as the conservative range, while the creator profile
    # may expose user-selected 1080p scene lengths through the 30s B200/DFR ceiling.
    # Final runtime is independent and may span multiple continuity-locked scenes.
    ltx_native_chunk_seconds: float = 20.0
    max_preview_scene_seconds: float = 20.0
    max_1080p_scene_seconds: float = 30.0
    max_4k_scene_seconds: float = 15.0
    max_factory_duration_seconds: int = 300
    factory_min_scene_seconds: float = 15.0
    factory_standard_max_scene_seconds: float = 20.0
    factory_enable_30s_1080p_single_pass: bool = True
    factory_experimental_1080p_scene_seconds: float = 30.0
    factory_allow_fallback_final: bool = False

    # Enterprise reusable Elements. References are workspace-scoped and immutable
    # version manifests are bound into jobs so later edits cannot silently change
    # an already-rendered project. The Ingredients IC-LoRA is the multi-element
    # identity path; first-frame mode remains available for exact image animation.
    element_max_stored_per_workspace: int = 100
    element_max_assets_per_element: int = 8
    element_max_active_per_scene: int = 6
    element_max_characters_per_scene: int = 3
    element_max_props_per_scene: int = 3
    element_max_locations_per_scene: int = 1
    element_max_styles_per_scene: int = 1
    element_max_upload_mb: int = 15
    # Signed, asset-scoped preview URLs let <img> tags work even when local
    # development uses a split web/API origin. They never expose the workspace token.
    element_asset_url_ttl_seconds: int = 21600
    element_reference_sheet_width: int = 768
    element_reference_sheet_height: int = 448
    element_ingredients_enabled: bool = True
    element_ingredients_strength: float = 1.0
    # Ingredients was trained at 121 frames. The Modal worker keeps longer creator
    # shots in one output while streaming 121-frame overlapping temporal windows,
    # so the explicit 30s 1080p creator profile can retain reference conditioning.
    element_ingredients_max_scene_seconds: float = 30.0

    # Final-render audio guard. Gemini inspects rendered audio for gibberish or
    # unintended speech; Modal can repair failed audio via LTX Retake while keeping
    # the picture frozen.
    factory_audio_qc_enabled: bool = True
    factory_audio_retake_enabled: bool = True
    factory_audio_qc_strict_final: bool = True
    # Compatibility flags retained for existing .env files. Successful media is
    # always retained, including on QC rejection, provider outage or repair failure.
    factory_qc_fail_open_on_unavailable: bool = True
    # Failed QC stays visible and prevents automatic publishing. Legacy false
    # values no longer authorize deleting successful renders.
    factory_preserve_on_qc_failure: bool = True

    # Continuity/cardinality guard. "auto" requests use this visual QC gate when
    # Gemini is configured; QC failures can trigger a bounded regeneration before
    # a scene is accepted into the factory timeline.
    continuity_vision_qc_enabled: bool = True
    continuity_qc_timeout_seconds: float = 12.0
    continuity_qc_max_frames: int = 5

    # Demo OTP can be displayed in any environment, including the hosted demo.
    # Set false to send codes by SMTP instead of returning them to the browser.
    auth_enabled: bool = True
    demo_auth_show_otp: bool = True
    auth_otp_ttl_seconds: int = 600
    auth_otp_max_attempts: int = 5
    auth_session_days: int = 30
    auth_otp_resend_seconds: int = 30
    auth_otp_requests_per_hour: int = 20
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from_email: str = ""
    smtp_use_tls: bool = True
    smtp_use_ssl: bool = False
    smtp_timeout_seconds: float = 15.0

    # Account-owned Studio history. Browser localStorage is only a cache; the
    # canonical previous-chat list lives in SQLite and follows the signed-in user.
    chat_history_limit: int = 100
    chat_workspace_max_bytes: int = 1_500_000

    # Workspace/session signing. Required when billing or YouTube integrations
    # are enabled in production. Never commit the production value.
    triven_secret_key: str = ""

    # Stripe Checkout + credit ledger. Billing can be wired and tested while
    # enforcement stays off; turn BILLING_ENFORCE_CREDITS=true only after the
    # live webhook has been verified end-to-end.
    billing_enabled: bool = False
    billing_enforce_credits: bool = False
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_price_starter: str = ""
    stripe_price_pro: str = ""
    stripe_price_studio: str = ""
    stripe_starter_seconds: int = 300
    stripe_pro_seconds: int = 1800
    stripe_studio_seconds: int = 7200

    # YouTube Data API OAuth. The server stores refresh tokens encrypted at
    # rest with a key derived from TRIVEN_SECRET_KEY.
    youtube_enabled: bool = False
    youtube_client_id: str = ""
    youtube_client_secret: str = ""
    youtube_redirect_uri: str = ""
    youtube_allow_public: bool = False

    # Single-VPS production safety. One worker avoids duplicate in-memory queues
    # and prevents accidental parallel paid GPU renders.
    job_workers: int = 1
    job_max_pending: int = 3
    job_retention_days: int = 14

    # Keep VPS local storage bounded. Preview clips should age out quickly;
    # final renders are kept longer by default.
    preview_retention_days: int = 3
    final_retention_days: int = 30
    minimum_free_disk_gb: float = 10.0
    metrics_max_bytes: int = 10 * 1024 * 1024
    log_max_bytes: int = 20 * 1024 * 1024
    ffmpeg_timeout_seconds: int = 900
    ffprobe_timeout_seconds: int = 30

    # Expensive synchronous render endpoints are convenient for local smoke tests
    # but bypass the bounded production job queue. Disable them on the public server.
    enable_sync_render_endpoints: bool = True
    enable_metrics_endpoint: bool = True

    @property
    def is_production(self) -> bool:
        return self.app_env.strip().lower() in {"production", "prod"}

    @property
    def youtube_callback_url(self) -> str:
        if self.youtube_redirect_uri.strip():
            return self.youtube_redirect_uri.strip()
        return f"{self.frontend_url.rstrip('/')}" + "/api/v1/youtube/callback"

    @property
    def cors_origin_list(self) -> list[str]:
        return [
            item.strip().rstrip("/")
            for item in self.cors_origins.split(",")
            if item.strip()
        ]

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
