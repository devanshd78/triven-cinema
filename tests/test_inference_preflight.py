import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from inference.providers.modal_ltx import ModalLTXProvider

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("triven_model_manifest", ROOT / "modal/models.py")
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)


class InferencePreflightTests(unittest.TestCase):
    def setUp(self):
        ModalLTXProvider._preflight_cache.clear()
        env = patch.dict("os.environ", {"TRIVEN_LTX_REPO_REF": "v1.4.2"})
        env.start()
        self.addCleanup(env.stop)

    def test_ingredients_does_not_require_unused_detailing_adapter(self):
        files = models.required_model_paths(render_mode="dfr", element_reference_required=True, decoder="diffusion")
        self.assertIn(models.INGREDIENTS_LORA, files)
        self.assertNotIn(models.DETAILING_LORA, files)
        self.assertNotIn(models.VIDEO_VAE_CONV, files)

    def test_retake_only_requires_its_models(self):
        files = models.required_model_paths(operation="retake_audio")
        self.assertIn(models.VIDEO_VAE_DIFFUSION, files)
        self.assertNotIn(models.SPATIAL_UPSCALER, files)

    def test_missing_weights_prevent_gpu_rpc(self):
        endpoint = Mock()
        endpoint.remote.return_value = {"protocol_version": 2, "ltx_repo_ref": "v1.4.2", "ready": False, "errors": ["Missing model weights"]}
        with patch("inference.providers.modal_ltx.modal.Function.from_name", return_value=endpoint) as lookup:
            with self.assertRaisesRegex(RuntimeError, "Missing model weights"):
                ModalLTXProvider().generate("presenter", 512, 512, 5, 42, "conv")
        self.assertEqual(lookup.call_count, 1)
        self.assertEqual(lookup.call_args.args[1], "preflight")

    def test_provider_saves_base_and_refined_video_separately(self):
        endpoint = Mock()
        endpoint.remote.return_value = {"video_bytes": b"refined", "base_video_bytes": b"base", "detail_refined": True}
        with tempfile.TemporaryDirectory() as tmp, patch("inference.providers.modal_ltx.GENERATED_DIR", Path(tmp)), \
             patch.object(ModalLTXProvider, "preflight", return_value={"ready": True}), \
             patch("inference.providers.modal_ltx.modal.Function.from_name", return_value=endpoint):
            result = ModalLTXProvider().generate("presenter", 512, 512, 5, 42, "conv")
            self.assertEqual(Path(result.path).read_bytes(), b"refined")
            self.assertEqual(Path(result.base_path).read_bytes(), b"base")
            self.assertNotEqual(result.path, result.base_path)

    def test_optional_base_save_failure_does_not_lose_main_video(self):
        endpoint = Mock()
        endpoint.remote.return_value = {"video_bytes": b"refined", "base_video_bytes": b"base", "detail_refined": True}
        with patch.object(ModalLTXProvider, "preflight", return_value={"ready": True}), \
             patch.object(ModalLTXProvider, "_write_video", side_effect=[Path("/tmp/refined.mp4"), OSError("Disk full")]), \
             patch("inference.providers.modal_ltx.modal.Function.from_name", return_value=endpoint):
            result = ModalLTXProvider().generate("presenter", 512, 512, 5, 42, "conv")
        self.assertEqual(result.filename, "refined.mp4")
        self.assertIsNone(result.base_path)

    def test_old_worker_is_rejected_before_gpu_rpc(self):
        endpoint = Mock()
        endpoint.remote.return_value = {"protocol_version": 1, "ready": True}
        with patch("inference.providers.modal_ltx.modal.Function.from_name", return_value=endpoint):
            with self.assertRaisesRegex(RuntimeError, "protocol is incompatible"):
                ModalLTXProvider().preflight()

    def test_auth_error_is_actionable(self):
        with patch("inference.providers.modal_ltx.modal.Function.from_name", side_effect=RuntimeError("Unauthenticated")):
            with self.assertRaisesRegex(RuntimeError, "Verify Modal authentication"):
                ModalLTXProvider().preflight()

    def test_disabled_workspace_stops_before_gpu_and_explains_account_recovery(self):
        endpoint = Mock()
        endpoint.remote.side_effect = RuntimeError('workspace ac-private-workspace is disabled')
        with patch('inference.providers.modal_ltx.modal.Function.from_name', return_value=endpoint) as lookup:
            with self.assertRaisesRegex(RuntimeError, 'Modal workspace is disabled') as error:
                ModalLTXProvider().generate('presenter', 512, 512, 5, 42, 'conv')
        self.assertEqual(lookup.call_count, 1)
        self.assertNotIn('ac-private-workspace', str(error.exception))
        self.assertNotIn('modal deploy', str(error.exception))

    def test_wrong_ltx_revision_rejected_before_gpu_rpc(self):
        endpoint = Mock()
        endpoint.remote.return_value = {"protocol_version": 2, "ltx_repo_ref": "v1.4.1", "ready": True}
        with patch("inference.providers.modal_ltx.modal.Function.from_name", return_value=endpoint) as lookup:
            with self.assertRaisesRegex(RuntimeError, "revision mismatch: expected v1.4.2, received v1.4.1"):
                ModalLTXProvider().generate("presenter", 512, 512, 5, 42, "conv")
        self.assertEqual(lookup.call_count, 1)

    def test_revision_configuration_is_part_of_cache_key(self):
        endpoint = Mock()
        endpoint.remote.return_value = {"protocol_version": 2, "ltx_repo_ref": "v1.4.2", "ready": True}
        with patch("inference.providers.modal_ltx.modal.Function.from_name", return_value=endpoint):
            ModalLTXProvider().preflight()
            with patch.dict("os.environ", {"TRIVEN_LTX_REPO_REF": "v1.4.3"}):
                with self.assertRaisesRegex(RuntimeError, "expected v1.4.3"):
                    ModalLTXProvider().preflight()
            self.assertEqual(endpoint.remote.call_count, 2)

    def test_preflight_checks_authenticated_weight_heads_without_download(self):
        files = [models.TRANSFORMER, models.INGREDIENTS_LORA]
        with (
            patch("huggingface_hub.get_hf_file_metadata", return_value=SimpleNamespace(size=None)) as head,
            patch("huggingface_hub.hf_hub_download") as download,
            patch("huggingface_hub.HfApi.model_info") as repository_info,
        ):
            self.assertEqual(models.check_model_file_access(files, "test-token"), [])
        self.assertEqual(head.call_count, 2)
        self.assertTrue(all(call.kwargs == {"token": "test-token", "timeout": 10} for call in head.call_args_list))
        urls = [call.args[0] for call in head.call_args_list]
        self.assertTrue(any("/Lightricks/LTX-2.5/resolve/main/diffusion_models/" in url for url in urls))
        self.assertTrue(any("/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients/resolve/main/" in url for url in urls))
        download.assert_not_called()
        repository_info.assert_not_called()

    def test_gated_weight_denial_fails_even_when_repository_metadata_is_public(self):
        with (
            patch("huggingface_hub.HfApi.model_info", return_value=Mock()),
            patch("huggingface_hub.get_hf_file_metadata", side_effect=PermissionError("gated file denied")),
        ):
            errors = models.check_model_file_access([models.TRANSFORMER], "test-token")
        self.assertEqual(len(errors), 1)
        self.assertIn("gated-model approval", errors[0])
        self.assertIn(models.TRANSFORMER.name, errors[0])
        self.assertNotIn("test-token", errors[0])

    def test_nonempty_but_truncated_local_weight_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "partial.safetensors"
            path.write_bytes(b"partial")
            with patch.object(models, "MODEL_ROOT", root), patch("huggingface_hub.get_hf_file_metadata", return_value=SimpleNamespace(size=100)):
                errors = models.check_model_file_access([path], "test-token")
        self.assertIn("size mismatch", errors[0])

    def test_successful_manifest_cached_per_recipe(self):
        endpoint = Mock()
        endpoint.remote.return_value = {"protocol_version": 2, "ltx_repo_ref": "v1.4.2", "ready": True}
        with patch("inference.providers.modal_ltx.modal.Function.from_name", return_value=endpoint):
            provider = ModalLTXProvider()
            provider.preflight()
            provider.preflight()
            provider.preflight(element_reference_required=True)
            self.assertEqual(endpoint.remote.call_count, 2)
            provider.preflight(force_refresh=True)
            self.assertEqual(endpoint.remote.call_count, 3)


if __name__ == "__main__":
    unittest.main()
