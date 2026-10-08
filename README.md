# Triven Cinema

Triven Cinema is an AI video factory: prompt input, continuity-locked storyboard planning, self-hosted LTX-2.5 on on-demand Modal GPU, long-form scene generation, synchronized/native audio handling, FFmpeg delivery mastering, Stripe generation credits, connected-channel YouTube publishing, persistent render jobs, progress, and cost/benchmark tooling.

## Implemented now

- Prompt -> Gemini storyboard -> editable scene prompts.
- Direct prompt mode that skips storyboard generation.
- Textual storyboard prompt-coverage diagnostic.
- LTX-2.5 through:
  - Hugging Face ZeroGPU development fallback.
  - Self-hosted Modal deployment with persistent model/output volumes.
- B200 deployment path already supported through `TRIVEN_MODAL_GPU`.
- 16:9, 9:16 and 1:1 source profiles.
- Customer duration profiles: preview up to 10s, 1080p up to 30s/scene, 4K delivery up to 15s/scene.
- LTX-2.5 native temporal-window rendering for long Modal scenes with carry/blend overlap.
- Prompt-to-finished-video Factory mode with strict scene continuity and up to 300s total runtime by default.
- Persistent asynchronous video jobs in `storage/jobs/jobs.sqlite3`.
- OTP login with signed HttpOnly account sessions. `DEMO_AUTH_SHOW_OTP=true` displays the code in the browser in development and production; SMTP is only needed when this flag is false.
- Account-owned Previous Chats in `storage/chats/chats.sqlite3`, with safe one-time migration from the old browser-only history.
- UI polling with coarse real job stages: queued -> initializing -> rendering -> delivery -> probing -> complete.
- Per-scene source previews and regeneration.
- Existing previews are reused when combining; missing scenes only are rendered.
- Storyboard scene previews stay at source resolution even when final quality is 1080p, so the final sequence is upscaled only once.
- FFmpeg scene composition with audio-preservation handling.
- Native/embedded audio stream probing through `ffprobe`.
- Native/mastered/mute audio delivery modes; mastered mode applies final loudness normalization.
- Stripe Checkout credit packs with idempotent webhook ledger, render charging, and failed-job refunds.
- Per-workspace encrypted YouTube OAuth connection and resumable final-master uploads.
- Final MP4 preview and download.
- Durable owned media, partial-job recovery, saved takes, and refresh-safe asynchronous generation/composition.
- Separate spoken scripts preserved through planning, per-scene visual/audio QC, and explicit audio-only repair with the original compressed picture preserved.
- Render timing metrics and optional GPU cost estimates.
- Seed, preview-decoder, prompt-enhancement and production realism controls in the UI.
- Creator-grade talking-head preset with stage-2 Character IC-LoRA lock, prompt-authoritative wardrobe, early/mid/late artifact QC, user-selected final runtime, and up to 30s per 1080p creator scene.
- Semantic Character reference roles (face/full body/profile/costume) and identity-only reference-sheet composition when the prompt requests different clothing.
- Modal GPU benchmark script and required-aspect-ratio smoke test.
- Docker deployment baseline for the Next.js/FastAPI application layer.

## Important quality wording

Final-quality Modal jobs now use the LTX production pipelines on a 1080-class source canvas (for example 1920x1088 at 16:9) before final delivery normalization/crop. 4K uses the validated LTX-aligned 3840x2176 source grid before UHD delivery. Preview mode remains lower resolution for cost/speed.

The UI does not treat resolution alone as a quality guarantee: Character identity, prompt adherence, temporal stability, audio correctness and artifact QC are separate production checks.

## Local setup

```bash
cp .env.example .env
```

Add `GEMINI_API_KEY` and `HF_TOKEN` locally. Never commit `.env`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r services/api/requirements.txt
cd apps/web && npm ci && cd ../..
```

Run API:

```bash
./scripts/run_api.sh
```

Run web in another terminal:

```bash
./scripts/run_web.sh
```

Open `http://localhost:3000`.

## Modal production path

Prerequisites:

1. Accept LTX-2.5 gated access on Hugging Face.
2. Authenticate Modal.
3. Enable the required paid GPU access.
4. Keep a valid `HF_TOKEN` in local `.env`.

Prepare and deploy:

```bash
./scripts/setup_modal.sh
```

Use:

```env
VIDEO_PROVIDER="modal"
TRIVEN_MODAL_GPU="B200"
```

The web UI now defaults to Modal production but still allows ZeroGPU development fallback.


## Real Skin / Identity Max production path

Final-quality Cinema Studio renders now expose three realism profiles:

- `Standard`: existing LTX production path without the extra refinement pass.
- `Real Skin` (Studio default): DFR for normal scenes, or Ingredients IC-LoRA when reusable identity Elements are active, with the diffusion video VAE, followed by Lightricks' official `LTX-2.5-22b-IC-LoRA-Refine-Details` tiled video-to-video pass. The refiner uses generic photographic-detail wording so each tile rebuilds texture without being told to repaint a specific face or subject.
- `Identity Max`: the same Real Skin final path, but it requires a Character Element in `Identity` mode so the base generation is conditioned by a sharp real reference photograph before the final tiled detail reconstruction.

The Refine Details adapter is not trained for audio generation. Triven therefore refines the picture, then remuxes the untouched audio stream from the original LTX scene so speech timing and synchronized sound are not replaced by the texture pass. Preview renders skip this expensive second pass.

Triven does not blindly feed the raw Character portrait into Refine Details at frame `-1`. That optional LTX reference is spatial/tile sensitive and should be aligned to the output canvas; an unaligned portrait can hurt identity consistency. Character references remain in the base identity-conditioning stage until subject-aware alignment is implemented.

Important: the official DFR pipeline still expects the **distilled** LTX-2.5 transformer; do not replace it with the full/dev transformer. The full/dev checkpoint is for other guided pipelines, not DFR.

The Refine Details repository is gated separately on Hugging Face. Before running `./scripts/setup_modal.sh`, accept access for both LTX-2.5 and `Lightricks/LTX-2.5-22b-IC-LoRA-Refine-Details`. The setup command downloads the refiner into the existing persistent Modal model volume.

The application does not claim a numeric face-match guarantee. Generative identity can still drift because of pose, occlusion, lighting, motion and reference quality; Identity Max is the strongest supported path, not a biometric guarantee.

See `docs/V11_CREATOR_GRADE_REALISM.md` for the creator-grade identity/wardrobe/QC architecture and recommended Character-reference workflow.

## Native audio verification

Every API generation response includes `media_info.has_audio` and `media_info.audio_codec` based on `ffprobe`.

For any existing MP4:

```bash
python scripts/inspect_media.py storage/generated/<video>.mp4
```

This verifies whether the actual output contains an audio stream; merely loading the LTX audio VAE is not treated as proof of audible output.

## Persistent asynchronous jobs

The UI uses:

```text
POST /api/v1/generations/jobs/video
GET  /api/v1/generations/jobs/{job_id}
```

Jobs are persisted in SQLite under `storage/jobs/`. If the API process restarts while a job is running, that interrupted job is marked failed instead of being left permanently "running".

The original synchronous endpoint remains available for CLI smoke tests:

```text
POST /api/v1/generations/video
```

## Cost tracking

Render timing is measured automatically. To enable cost estimates, copy the **current** hourly rates from your Modal dashboard into `.env`:

```env
MODAL_GPU_HOURLY_USD_B200=0
MODAL_GPU_HOURLY_USD_H200=0
MODAL_GPU_HOURLY_USD_H100=0
```

Do not enter guessed rates. When configured, responses include:

- `estimated_cost_usd`
- `estimated_cost_per_output_minute_usd`
- an explicit note that these are estimates, not billed cost

Metrics summary:

```bash
curl http://127.0.0.1:8000/api/v1/generations/metrics/summary
```

## Paid GPU benchmark tooling

The scripts refuse to run unless you explicitly opt in.

B200/H200/H100 comparison:

```bash
RUN_PAID_BENCHMARKS=1 ./scripts/benchmark_gpus.sh
```

Required aspect-ratio smoke test:

```bash
RUN_PAID_BENCHMARKS=1 python scripts/test_modal_aspects.py
```

Results are written under `storage/benchmarks/`.

## Verification

```bash
./scripts/verify_mvp.sh
```

It checks Python syntax, unit tests, shell syntax and the frontend build when dependencies are installed.

## Application-layer deployment baseline

The production target is a **Hostinger Linux VPS** running the application layer with Docker Compose, while LTX inference remains on **Modal GPU**. See `deploy/hostinger/README.md`.

```bash
./scripts/deploy_hostinger.sh
```

The browser uses same-origin `/api` and `/media` routes through the existing host Nginx. FastAPI and Next.js bind only to loopback ports 3334/3333 and are not published directly to the internet.

For multi-instance production, move generated media/job state from local disk to object storage/Postgres before scaling horizontally.

## Main API endpoints

```text
GET  /api/v1/health
GET  /api/v1/generations/capabilities
POST /api/v1/generations/plan
POST /api/v1/generations/video
POST /api/v1/generations/jobs/video
GET  /api/v1/generations/jobs/{job_id}
POST /api/v1/generations/combine
POST /api/v1/generations/full-video
GET  /api/v1/generations/metrics/summary
GET  /api/v1/generations/download/{filename}
POST /api/v1/factory/jobs
GET  /api/v1/billing/catalog
GET  /api/v1/billing/me
POST /api/v1/billing/checkout
POST /api/v1/billing/webhook
GET  /api/v1/youtube/status
POST /api/v1/youtube/connect
POST /api/v1/youtube/publish
```

## Still not honestly "done"

These require additional services or measured validation:

- WAN inference implementation/version selection.
- MiniMax inference implementation/version selection.
- Native/high-fidelity 1080p LTX benchmark and optimal inference/pass settings.
- Actual H100/H200/B200 cost comparison until the paid benchmark scripts are run.
- Continuous temporal/lip-sync/voice-identity evaluation beyond sampled visual frames and semantic audio inspection.
- ElevenLabs voice generation and voice/video synchronization.
- Background music generation/mixing.
- Latest-topic research -> script automation.
- Production Postgres/object storage/HA deployment.

---

## Hostinger VPS production server

The current production target is **one Hostinger Linux VPS for the application/orchestration layer** and **Modal for LTX GPU inference**. The large LTX model remains on Modal and is not loaded on the VPS.

The production path is:

```text
Internet / HTTPS
      ↓
Host Nginx on Hostinger VPS · ports 80/443
      ↓
  ┌───────────────┬────────────────┐
  ↓               ↓                ↓
Next.js :3000   FastAPI :8000    /media
                  ↓
     SQLite jobs + generated media
                  ↓
         Gemini/fallback planner
                  ↓
             Modal GPU / LTX 2.5
```

Production behavior:

- Host Nginx is the only public-facing HTTP service for this deployment.
- FastAPI and Next.js stay on the private Docker network.
- `/api/*` and `/media/*` are routed by host Nginx to FastAPI on 127.0.0.1:3334; all other traffic goes to Next.js on 127.0.0.1:3333.
- Modal authentication uses `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in the VPS `.env`.
- Paid render jobs remain bounded (`JOB_WORKERS=1`, `JOB_MAX_PENDING=3` by default).
- Generated media is served through a signed-workspace ownership check; jobs SQLite, billing/integration state, metrics and backups stay private.
- Health checks, Docker restart policies and log rotation are enabled.
- Generated previews/finals are retained according to `.env` limits and cleaned by the maintenance container.
- SQLite/metrics state is backed up before deployments and periodically by the maintenance container.
- Production exception responses are sanitized and API docs are disabled.

Start with:

```bash
cp .env.production.example .env
chmod 600 .env
# Fill TRIVEN_DOMAIN, Gemini/Modal credentials and provider settings.
python3 scripts/production_preflight.py
./scripts/deploy_hostinger.sh
./scripts/status_hostinger.sh
```

Full setup, DNS, firewall, deployment, logs, updates and backup guidance:

```text
deploy/hostinger/README.md
```

**Do not publish ports 3000 or 8000 on the VPS firewall.** Only SSH, HTTP and HTTPS should be publicly reachable.

### Reliability update deployment

The reliability update requires the new Modal worker (CPU preflight protocol 2). Hosted demo login uses `AUTH_ENABLED=true` and `DEMO_AUTH_SHOW_OTP=true`; no SMTP configuration is needed. Existing servers must update this flag in their `.env`, since changing the example file does not change a deployed environment. Keep the existing `TRIVEN_SECRET_KEY` (at least 32 characters) to preserve sessions and integrations. Demo mode displays the code to the person entering the email address; it does not verify mailbox ownership. Set the flag to false and configure SMTP if email verification is desired.

Run `modal deploy modal/app.py` separately before application rollout. `scripts/check_inference.py --all` checks enabled model recipes and authenticated weight access without allocating a GPU. The application deployment now stops on a failed backup, inference preflight, or public HTTPS health check.

Runtime databases and Character uploads are no longer tracked in Git. **Before the first VPS pull of this update, follow the one-time storage migration in `deploy/hostinger/README.md`.** A pull that removes formerly tracked files can otherwise remove the server's copies. Existing database schemas and media ownership records migrate automatically at API startup; later job pruning does not revoke asset ownership.

`./scripts/run_tests.sh` runs tests in an isolated temporary copy with empty runtime state. `./scripts/verify_mvp.sh` also checks syntax, frontend regression tests, lint and the production build. Tests do not establish real GPU image quality; validate speech, identity, transitions and runtime with representative renders after deploying.

## Generation latency

Generation runs sequentially when scenes depend on the previous approved frame. Real Skin and Identity Max add a separate texture refinement pass; failed visual QC can render another candidate, and failed audio QC can add an audio-only GPU repair. These operations can dominate runtime, so a long render is not necessarily a queue delay.

Quality inspections use one attempt per configured Gemini model within `QUALITY_CHECK_RETRY_BUDGET_SECONDS` (45 seconds by default). Planner retries share `GEMINI_RETRY_BUDGET_SECONDS` (90 seconds). HTTP timeouts shrink to the remaining retry budget; expired inspections report QC unavailable and preserve the clip. They do not approve unchecked video. Generation and refinement invoke the image's preinstalled Python environment directly without repeating `uv run` dependency resolution. Worker logs and saved attempt metadata separate base generation, refinement, readiness, and queue/startup/transfer time.

Composition copies a single compatible MP4 unchanged and joins compatible H.264 scene pictures without another video encode. Codec headers, dimensions, frame rate, time base, and stream layout must match; incompatible inputs or a failed stream copy use the encoding fallback. Delivery still applies the requested final dimensions, and joined audio is normalized as before.

For a faster draft, select Preview and Standard in the existing controls; Character references still apply. Select final resolution and the desired realism profile when preparing the final take. No preset or latency optimization silently lowers the chosen resolution, skips identity conditioning, or treats unavailable QC as passed.

API retry/progress changes require an API rebuild; worker startup/timing changes require `modal deploy modal/app.py`. GPU speedup must be measured with comparable real renders after readiness passes.

## Presenter reference and prompt validation

Character conditioning now uses one portrait per character rather than combining several photos of the same person into a collage. With **Follow scene prompt**, a labeled face takes priority over a profile and the scene must describe the desired clothes explicitly. With **Lock reference outfit**, the selected primary photo supplies both the person and the outfit. Other uploads remain saved. Visual QC uses the same selected identity reference as generation.

Put spoken words in **Spoken dialogue**, including Hindi, and ambience/voice direction in **Sound direction**. Unquoted narration pasted into a visual prompt is not automatically treated as speech. Hindi sentence punctuation is supported when distributing dialogue across storyboard scenes.

The Modal worker preserves the original take before successful texture refinement. When the refined video fails visual QC, the API checks that original before requesting another GPU render. It selects the original only on a passing QC verdict; failed and unavailable checks cannot approve it. Both takes remain available as generated assets, and cost accounting includes all work already performed. This requires the updated worker and API together; old workers remain compatible but cannot return the original take.

For a small initial presenter test, use one 10-second shot, 16:9, 1080p, Standard, Identity / Element reference, and Follow scene prompt. Keep one clear face photo labeled `face`; avoid group photos as the active portrait. Use an explicit outfit description instead of asking the renderer to infer a costume filename. For example:

**Visual prompt**

> A single continuous medium shot of @char3 presenting one matte-black smartphone in a modern technology studio. He stands behind a wooden desk, wearing a plain black long-sleeved button-up shirt with an open collar. He looks into the camera and speaks with a friendly expression, holding the phone upright in his right hand at chest height. His left hand makes one small explanatory gesture, then rests on the desk. A soft key light illuminates his face evenly; the neutral studio background stays gently out of focus. The eye-level camera remains still, with the presenter, his hands and the phone clearly visible throughout. The picture is clean and full-frame, with one presenter, one phone, and an unobstructed view.

**Spoken dialogue**

> नमस्कार दोस्तों! आज हम इस स्मार्टफोन की स्क्रीन और कैमरे को करीब से देखेंगे।

**Sound direction**

> One warm, conversational Hindi male voice, clear speech synchronized with the visible presenter, and quiet studio room tone. No background music or additional voices.

This is a validation recipe, not a guarantee of matching a real person's face or voice. Check the first, middle and last frames for a physical presenter in the studio, stable identity, one phone in the right hand, coherent clothing, and clean imagery. Listen for the exact line without repetitions. Only after that passes should the second camera angle and longer script be added. [LTX's prompting guide](https://docs.ltx.io/open-source-model/usage-guides/prompting-guide) recommends a focused shot with concrete action and clear camera/audio direction; [Ingredients documentation](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients) describes the separate reference/target structure used internally.

## Character continuity for multi-scene stories

Storyboard mode now defaults to **Strict continuity**. Triven creates one shared character bible and one shared visual-style bible, injects the same immutable identity locks into every scene prompt, and keeps the same seed across the whole story. After each rendered scene, FastAPI extracts the final video frame. The next Modal/LTX-2.5 render receives that PNG as frame-0 image conditioning, so the next clip starts from the actual visual identity produced by the previous clip instead of reinterpreting the character from text alone.

Continuity modes:

- `strict`: shared identity/style locks + same seed + previous-scene first-frame conditioning on Modal.
- `balanced`: shared identity/style locks + same seed, without image conditioning.
- `off`: independent scene generation.

For a 45-second demo, use **9 scenes x 5 seconds**. The UI supports up to 10 storyboard scenes.

The demo production domain is currently pinned to **https://devansh.info**. Host Nginx, production environment examples, preflight checks, and Hostinger health checks all target that hostname.


## AI video factory / billing / YouTube

See `docs/AI_VIDEO_FACTORY.md` for the long-form LTX profile, audio pipeline, Stripe customer-payment flow, connected YouTube publishing, and Hostinger deployment instructions.

## Continuity & cardinality engine

Strict story continuity now combines character/style bibles, physical entity-count locks, lossless first-frame conditioning, stable near-end anchor selection, long-shot temporal carry, vision QC and bounded auto-regeneration. See `docs/CONTINUITY_ENGINE.md` for the production behavior and failure-handling rules.

### Reusable Elements / Cinema Studio workflow

The web studio supports reusable characters, props, locations and styles with `@mentions`, protected multi-image references, immutable Element versions, exact-start-frame or identity-reference modes, and a simplified Cinema Studio-style Director panel. Identity guides are generated at the target duration/resolution with a 121-frame minimum, and supporting reference views are included in the Element sheet instead of being ignored. See `docs/AI_VIDEO_FACTORY.md` and `docs/CONTINUITY_ENGINE.md` for the generation mapping.
