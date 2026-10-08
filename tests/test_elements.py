import io
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

from fastapi.testclient import TestClient

from PIL import Image

from app.main import app
from app.schemas.elements import ElementBinding
from app.services import element_service
from app.services.element_service import (
    ElementError,
    UploadedElementAsset,
    build_reference_sheet,
    compile_element_prompt,
    create_element,
    elements_for_scene,
    get_element,
    list_elements,
    resolve_element_bindings,
)


def image_bytes(color: tuple[int, int, int], size=(512, 512)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


class ElementServiceTests(unittest.TestCase):
    def setUp(self):
        self.workspace = uuid.uuid4().hex
        element_service.initialize_element_store()

    def tearDown(self):
        with element_service._connect() as conn:  # test-only cleanup
            conn.execute("DELETE FROM element_versions WHERE workspace_id=?", (self.workspace,))
            conn.execute("DELETE FROM element_assets WHERE workspace_id=?", (self.workspace,))
            conn.execute("DELETE FROM elements WHERE workspace_id=?", (self.workspace,))
            conn.commit()
        shutil.rmtree(element_service.ASSET_DIR / self.workspace, ignore_errors=True)

    def test_create_version_and_workspace_isolation(self):
        element = create_element(
            self.workspace,
            name="Radha",
            handle="Radha",
            element_type="character",
            description="Rose-pink lehenga, long black hair.",
            uploads=[UploadedElementAsset("radha.png", "image/png", image_bytes((220, 130, 160)))],
        )
        self.assertEqual(element["handle"], "Radha")
        self.assertEqual(element["current_version"], 1)
        self.assertEqual(len(element["assets"]), 1)
        self.assertEqual(len(list_elements(self.workspace)), 1)
        with self.assertRaises(ElementError):
            get_element(uuid.uuid4().hex, element["id"])

    def test_bindings_are_immutable_versioned_and_mention_scoped(self):
        radha = create_element(
            self.workspace,
            name="Radha",
            handle="Radha",
            element_type="character",
            description="Canonical Radha.",
            uploads=[UploadedElementAsset("radha.png", "image/png", image_bytes((210, 150, 160)))],
        )
        flute = create_element(
            self.workspace,
            name="Krishna Flute",
            handle="Flute",
            element_type="prop",
            description="Wooden bansuri.",
            uploads=[UploadedElementAsset("flute.png", "image/png", image_bytes((160, 110, 60), (640, 256)))],
        )
        bindings = resolve_element_bindings(
            self.workspace,
            [
                ElementBinding(element_id=radha["id"], version_id=radha["current_version_id"], handle="Radha", apply_to_all_scenes=True),
                ElementBinding(element_id=flute["id"], version_id=flute["current_version_id"], handle="Flute"),
            ],
        )
        scene = elements_for_scene("@Radha hears a distant melody.", bindings)
        self.assertEqual([item.handle for item in scene], ["Radha"])
        scene = elements_for_scene("@Radha reaches for @Flute.", bindings)
        self.assertEqual({item.handle for item in scene}, {"Radha", "Flute"})
        compiled = compile_element_prompt("@Radha reaches for @Flute.", scene)
        self.assertIn("Reference sheet:", compiled)
        self.assertIn("Generated video:", compiled)
        self.assertNotIn("@Radha", compiled)


    def test_start_frame_prompt_animates_immediately_without_fake_reference_sheet(self):
        radha = create_element(
            self.workspace,
            name="Radha",
            handle="Radha",
            element_type="character",
            description="Canonical Radha.",
            uploads=[UploadedElementAsset("radha.png", "image/png", image_bytes((210, 150, 160)))],
        )
        bindings = resolve_element_bindings(
            self.workspace,
            [
                ElementBinding(
                    element_id=radha["id"],
                    version_id=radha["current_version_id"],
                    handle="Radha",
                    reference_mode="start_frame",
                )
            ],
        )
        compiled = compile_element_prompt("@Radha slowly turns toward camera.", bindings)
        self.assertNotIn("Reference sheet:", compiled)
        self.assertIn("START FRAME BEHAVIOR", compiled)
        self.assertIn("Animate forward", compiled)
        self.assertIn("Generated video:", compiled)

    def test_reference_wardrobe_uses_one_selected_portrait_without_a_collage(self):
        radha = create_element(
            self.workspace,
            name="Radha",
            handle="Radha",
            element_type="character",
            description="Canonical Radha.",
            uploads=[
                UploadedElementAsset("front.png", "image/png", image_bytes((255, 0, 0), (300, 600)), role="face"),
                UploadedElementAsset("profile.png", "image/png", image_bytes((0, 255, 0), (300, 600)), role="profile"),
                UploadedElementAsset("body.png", "image/png", image_bytes((0, 0, 255), (300, 600)), role="full_body"),
            ],
        )
        bindings = resolve_element_bindings(
            self.workspace,
            [
                ElementBinding(
                    element_id=radha["id"],
                    version_id=radha["current_version_id"],
                    handle="Radha",
                    wardrobe_policy="reference",
                )
            ],
        )
        self.assertEqual(len(bindings[0].reference_asset_paths), 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = build_reference_sheet(bindings, Path(tmp) / "sheet.png")
            with Image.open(path).convert("RGB") as image:
                colors = {color for _, color in (image.getcolors(maxcolors=image.width * image.height) or [])}
                self.assertIn((255, 0, 0), colors)
                self.assertNotIn((0, 255, 0), colors)
                self.assertNotIn((0, 0, 255), colors)

    def test_face_role_wins_over_first_uploaded_profile_for_identity_and_qc(self):
        element = create_element(
            self.workspace, name="Presenter", handle="char3", element_type="character",
            description="Short black hair and a full dark beard.",
            uploads=[
                UploadedElementAsset("profile.png", "image/png", image_bytes((0, 255, 0)), role="profile"),
                UploadedElementAsset("face.png", "image/png", image_bytes((255, 0, 0)), role="face"),
                UploadedElementAsset("costume.png", "image/png", image_bytes((0, 0, 255)), role="costume"),
            ],
        )
        profile = next(asset for asset in element["assets"] if asset["role"] == "profile")
        element_service.update_element(self.workspace, element["id"], {"primary_asset_id": profile["id"]})
        binding = ElementBinding(element_id=element["id"], handle="char3", wardrobe_policy="prompt")
        resolved = resolve_element_bindings(self.workspace, [binding])
        self.assertEqual(resolved[0].reference_asset_roles[0], "profile")
        with tempfile.TemporaryDirectory() as tmp:
            sheet = build_reference_sheet(resolved, Path(tmp) / "sheet.png")
            with Image.open(sheet).convert("RGB") as image:
                colors = {color for _, color in image.getcolors(image.width * image.height)}
                self.assertIn((255, 0, 0), colors)
                self.assertNotIn((0, 255, 0), colors)
                self.assertNotIn((0, 0, 255), colors)
        qc_path = element_service.canonical_reference_paths(resolved)[0][1]
        self.assertEqual(str(qc_path), resolved[0].reference_asset_paths[1])

        # Explicit start-frame selection continues to honor the chosen primary.
        start = resolve_element_bindings(self.workspace, [binding.model_copy(update={"reference_mode": "start_frame"})])
        self.assertEqual(str(element_service.canonical_reference_paths(start)[0][1]), start[0].primary_asset_path)

    def test_reference_wardrobe_honors_primary_costume_photo(self):
        element = create_element(
            self.workspace, name="Presenter", handle="char3", element_type="character", description="Full beard.",
            uploads=[
                UploadedElementAsset("face.png", "image/png", image_bytes((255, 0, 0)), role="face"),
                UploadedElementAsset("costume.png", "image/png", image_bytes((0, 0, 255)), role="costume"),
            ],
        )
        costume = next(asset for asset in element["assets"] if asset["role"] == "costume")
        element_service.update_element(self.workspace, element["id"], {"primary_asset_id": costume["id"]})
        resolved = resolve_element_bindings(self.workspace, [ElementBinding(
            element_id=element["id"], handle="char3", wardrobe_policy="reference",
        )])
        with tempfile.TemporaryDirectory() as tmp:
            sheet = build_reference_sheet(resolved, Path(tmp) / "sheet.png")
            with Image.open(sheet).convert("RGB") as image:
                colors = {color for _, color in image.getcolors(image.width * image.height)}
                self.assertIn((0, 0, 255), colors)
                self.assertNotIn((255, 0, 0), colors)


    def test_character_uploads_receive_semantic_roles_in_order(self):
        element = create_element(
            self.workspace,
            name="Presenter",
            handle="Presenter",
            element_type="character",
            description="Canonical presenter.",
            uploads=[
                UploadedElementAsset("one.png", "image/png", image_bytes((200, 10, 10))),
                UploadedElementAsset("two.png", "image/png", image_bytes((10, 200, 10))),
                UploadedElementAsset("three.png", "image/png", image_bytes((10, 10, 200))),
                UploadedElementAsset("four.png", "image/png", image_bytes((100, 100, 100))),
            ],
        )
        self.assertEqual([asset["role"] for asset in element["assets"]], ["face", "full_body", "profile", "costume"])

    def test_prompt_wardrobe_uses_identity_views_and_drops_reference_costume_language(self):
        presenter = create_element(
            self.workspace,
            name="Presenter",
            handle="Presenter",
            element_type="character",
            description=(
                "Long dark-brown hair, warm brown eyes, oval face. "
                "She is wearing a black leather jacket and blue jeans. "
                "Natural freckles across the cheeks."
            ),
            uploads=[
                UploadedElementAsset("face.png", "image/png", image_bytes((255, 0, 0), (300, 600)), role="face"),
                UploadedElementAsset("body.png", "image/png", image_bytes((0, 255, 0), (300, 600)), role="full_body"),
                UploadedElementAsset("profile.png", "image/png", image_bytes((0, 0, 255), (300, 600)), role="profile"),
            ],
        )
        bindings = resolve_element_bindings(
            self.workspace,
            [
                ElementBinding(
                    element_id=presenter["id"],
                    version_id=presenter["current_version_id"],
                    handle="Presenter",
                    wardrobe_policy="prompt",
                )
            ],
        )
        compiled = compile_element_prompt(
            "@Presenter wears a white and soft-lavender cable-knit sweater.",
            bindings,
        )
        self.assertIn("Generated video: Presenter wears a white and soft-lavender cable-knit sweater.", compiled)
        self.assertIn("generated video wardrobe", compiled.lower())
        self.assertNotIn("black leather jacket", compiled.lower())
        self.assertNotIn("blue jeans", compiled.lower())
        with tempfile.TemporaryDirectory() as tmp:
            path = build_reference_sheet(bindings, Path(tmp) / "sheet.png")
            with Image.open(path).convert("RGB") as image:
                colors = {color for _, color in (image.getcolors(maxcolors=image.width * image.height) or [])}
                self.assertIn((255, 0, 0), colors)
                # A solo presenter must use ONE clean face portrait rather than
                # a split-screen face/profile contact sheet for IC-LoRA.
                self.assertNotIn((0, 0, 255), colors)
                self.assertNotIn((0, 255, 0), colors)

    def test_signed_asset_url_loads_without_workspace_header_or_cookie(self):
        element = create_element(
            self.workspace,
            name="Radha",
            handle="Radha",
            element_type="character",
            description="Canonical Radha.",
            uploads=[UploadedElementAsset("radha.png", "image/png", image_bytes((210, 150, 160)))],
        )
        asset_url = element["assets"][0]["asset_url"]
        self.assertIn("?access=", asset_url)

        with TestClient(app) as client:
            response = client.get(asset_url)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers.get("content-type"), "image/png")
            self.assertTrue(response.content.startswith(b"\x89PNG"))

            parts = urlsplit(asset_url)
            query = parse_qs(parts.query)
            token = query["access"][0]
            query["access"] = [token[:-1] + ("A" if token[-1] != "A" else "B")]
            tampered = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query, doseq=True), parts.fragment))
            denied = client.get(tampered)
            self.assertEqual(denied.status_code, 404)

    def test_reference_sheet_is_clean_composite(self):
        a = create_element(
            self.workspace,
            name="Radha", handle="Radha", element_type="character", description="",
            uploads=[UploadedElementAsset("a.png", "image/png", image_bytes((255, 0, 0), (300, 600)))],
        )
        b = create_element(
            self.workspace,
            name="Flute", handle="Flute", element_type="prop", description="",
            uploads=[UploadedElementAsset("b.png", "image/png", image_bytes((0, 255, 0), (800, 200)))],
        )
        bindings = resolve_element_bindings(
            self.workspace,
            [
                ElementBinding(element_id=a["id"], version_id=a["current_version_id"], handle="Radha"),
                ElementBinding(element_id=b["id"], version_id=b["current_version_id"], handle="Flute"),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = build_reference_sheet(bindings, Path(tmp) / "sheet.png")
            with Image.open(path) as image:
                self.assertEqual(image.size, (768, 448))
                self.assertEqual(image.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
