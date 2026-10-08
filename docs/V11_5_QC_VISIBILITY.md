# Video QC verdicts in every Triven Cinema output (v11.5)

Each **successfully rendered** video now has a QC report attached to its saved chat-history record. The UI displays **Visual / identity QC** and **Speech / audio QC** separately, including the actual reason when available.

Statuses are explicit and are never inferred from the presence of an MP4:

- **Passed:** the configured inspector finished and did not report a violation in the accepted final render.
- **Failed — review needed:** the inspector detected a problem after the retry budget; keep the MP4 for review and block automatic YouTube publishing.
- **Unavailable:** the inspector could not complete (for example a Gemini 429). Keep the MP4 but do not call it approved or auto-publish it.
- **Not checked:** the associated inspection did not run (muted audio, disabled QC or unsupported mode). Not equivalent to passing QC.
- **Not recorded:** a pre-v11.5 saved video lacks an authoritative verdict. It is not silently relabeled as passed.

Factory: visual and semantic audio QC use backend-reported states; warnings and retry counts remain visible beneath the finished video and in saved chats. A failed or unavailable QC blocks automatic YouTube publishing.

Direct: visual QC runs after generation even with `continuity_mode=off`, so inspecting a clip does not require adding verbose continuity instructions to its native speech prompt. A skipped/unavailable inspector or rejected clip does not discard the expensive rendered MP4 under the current production fail-open/preserve settings. Semantic audio QC is explicitly marked **Not checked** because the Direct pipeline does not run the Factory audio inspector.

Storyboard: each rendered scene shows its visual verdict and warnings. Combining scenes preserves the per-scene verdicts in the final video; any failed scene makes the final visual verdict **Failed**, and any unavailable scene (without a failure) makes it **Unavailable**. Scene combining does not re-run semantic audio QC, so its audio verdict is **Not checked**.

The verdict represents an automated inspection of sampled frames/audio, **not** a guarantee that the video has no issues. Previously generated clips cannot be retroactively inspected by merely upgrading the UI.
