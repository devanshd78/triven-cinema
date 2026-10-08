import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.long_render_service import render_long_clip, split_duration, split_duration_balanced
from inference.providers.base import VideoGenerationResult, VideoProvider


class _LegacyIngredientsModalProvider(VideoProvider):
    name = "modal-test"
    supports_native_long_video = True

    def __init__(self, root: Path):
        self.root = root
        self.calls: list[dict] = []

    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        duration_seconds: float,
        seed: int,
        decoder: str,
        enhance_prompt: bool = False,
        render_mode: str = "distilled",
        reference_image_path: str | None = None,
        reference_strength: float = 0.95,
        element_reference_sheet_path: str | None = None,
        element_reference_strength: float = 1.0,
        realism_profile: str = "standard",
    ) -> VideoGenerationResult:
        self.calls.append(
            {
                "duration": float(duration_seconds),
                "prompt": prompt,
                "reference": reference_image_path,
                "element_sheet": element_reference_sheet_path,
            }
        )
        if element_reference_sheet_path and duration_seconds > 20:
            raise RuntimeError(
                "Modal LTX generation failed. Remote error: Element identity-conditioned scenes "
                "are limited to 20s until the Ingredients profile is benchmarked beyond its training bucket."
            )
        path = self.root / f"chunk-{len(self.calls)}.mp4"
        path.write_bytes(b"video")
        return VideoGenerationResult(
            filename=path.name,
            path=str(path),
            seed=seed,
            render_details=f"{duration_seconds}s test chunk",
            render_seconds=1.0,
            prompt=prompt,
            provider=self.name,
            gpu="test-gpu",
            wall_seconds=1.0,
            reference_conditioned=bool(reference_image_path or element_reference_sheet_path),
            chunk_count=1,
            render_mode=render_mode,
            realism_profile=realism_profile,
        )


class LongRenderServiceTests(unittest.TestCase):
    def test_completed_chunks_survive_and_are_registered_when_later_render_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = _LegacyIngredientsModalProvider(root)
            original = provider.generate
            def fail_second_chunk(**kwargs):
                if len(provider.calls) >= 2:
                    raise RuntimeError("second chunk failed")
                return original(**kwargs)
            provider.generate = fail_second_chunk
            with (
                patch("app.services.long_render_service.GENERATED_DIR", root),
                patch("app.services.long_render_service.register_current_generated_asset") as register,
                patch("app.services.long_render_service.extract_last_frame", side_effect=lambda _source, target: target.write_bytes(b"frame")),
            ):
                with self.assertRaisesRegex(RuntimeError, "second chunk failed"):
                    render_long_clip(provider=provider, prompt="presenter", width=512, height=512,
                                     duration_seconds=30, seed=42, decoder="conv", element_reference_sheet_path="sheet.png")
                self.assertEqual(register.call_args.args[0], "chunk-2.mp4")
                self.assertTrue((root / "chunk-2.mp4").exists())

    def test_thirty_seconds_uses_three_ten_second_chunks(self):
        self.assertEqual(split_duration(30, 10), [10.0, 10.0, 10.0])

    def test_fifteen_seconds_uses_ten_plus_five(self):
        self.assertEqual(split_duration(15, 10), [10.0, 5.0])

    def test_short_clip_stays_single_chunk(self):
        self.assertEqual(split_duration(5, 10), [5.0])

    def test_balanced_legacy_fallback_avoids_short_tail(self):
        self.assertEqual(split_duration_balanced(30, 20), [15.0, 15.0])
        self.assertEqual(split_duration_balanced(25, 20), [12.5, 12.5])

    def test_identity_runtime_falls_back_when_modal_worker_still_has_legacy_20s_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            element_sheet = root / "sheet.png"
            element_sheet.write_bytes(b"sheet")
            provider = _LegacyIngredientsModalProvider(root)

            def fake_extract_last_frame(_video: Path, destination: Path) -> None:
                destination.write_bytes(b"frame")

            def fake_combine(_videos: list[Path], destination: Path) -> None:
                destination.write_bytes(b"combined")

            with (
                patch("app.services.long_render_service.GENERATED_DIR", root),
                patch("app.services.long_render_service.extract_last_frame", side_effect=fake_extract_last_frame),
                patch("app.services.long_render_service.combine_videos", side_effect=fake_combine),
            ):
                result = render_long_clip(
                    provider=provider,
                    prompt="One continuous creator shot with the same presenter.",
                    width=1920,
                    height=1088,
                    duration_seconds=30,
                    seed=42,
                    decoder="diffusion",
                    render_mode="dfr",
                    element_reference_sheet_path=str(element_sheet),
                    element_reference_strength=1.0,
                    realism_profile="identity_max",
                )

            # First try preserves the best path: one native 30s invocation. The
            # legacy worker rejects immediately, so Triven transparently retries
            # as two equal identity-locked continuation chunks.
            self.assertEqual([call["duration"] for call in provider.calls], [30.0, 15.0, 15.0])
            self.assertTrue(all(call["element_sheet"] == str(element_sheet) for call in provider.calls))
            self.assertIsNone(provider.calls[1]["reference"])
            self.assertIsNotNone(provider.calls[2]["reference"])
            self.assertIn("CONTINUATION PART 2 OF 2", provider.calls[2]["prompt"])
            self.assertEqual(result.chunk_count, 2)
            self.assertTrue(Path(result.path).exists())


if __name__ == "__main__":
    unittest.main()
