import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.api.routes import elements as routes
from app.services import element_service as service
from app.schemas.elements import ElementBinding


def upload(name='face.png', color='red', role='face', size=(256, 320)):
    output = io.BytesIO()
    Image.new('RGB', size, color).save(output, format='PNG')
    return service.UploadedElementAsset(name, 'image/png', output.getvalue(), role)


class ElementUploadRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.paths = patch.multiple(service, ELEMENT_STORAGE_DIR=root, ASSET_DIR=root / 'assets', DB_PATH=root / 'elements.sqlite3')
        self.paths.start()
        self.workspace = 'a' * 32
        service.initialize_element_store()

    def tearDown(self):
        self.paths.stop()
        self.temp.cleanup()

    def create(self, uploads=None, **kwargs):
        return service.create_element(self.workspace, name='Actor', handle='Actor', element_type='character',
                                      description='Same face', uploads=uploads or [upload()], **kwargs)

    def test_invalid_batch_names_file_and_leaves_no_partial_save(self):
        with self.assertRaisesRegex(service.ElementError, 'broken.png'):
            self.create([upload(), service.UploadedElementAsset('broken.png', 'image/png', b'broken')])
        self.assertEqual(service.list_elements(self.workspace), [])
        self.assertEqual(list(service.ASSET_DIR.rglob('*.png')), [])

    def test_failed_add_preserves_original_assets_and_version(self):
        element = self.create()
        before = list(service.ASSET_DIR.rglob('*.png'))
        with self.assertRaisesRegex(service.ElementError, 'tiny.png'):
            service.add_element_assets(self.workspace, element['id'], [upload('valid.png'), upload('tiny.png', size=(20, 20))])
        saved = service.get_element(self.workspace, element['id'])
        self.assertEqual(saved['current_version'], 1)
        self.assertEqual(list(service.ASSET_DIR.rglob('*.png')), before)

    def test_failure_during_second_file_write_cleans_first_file(self):
        element = self.create()
        before = set(service.ASSET_DIR.rglob('*.png'))
        save = service._save_asset
        calls = 0
        def failing_save(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError('disk unavailable')
            return save(*args, **kwargs)
        with patch.object(service, '_save_asset', side_effect=failing_save), self.assertRaises(OSError):
            service.add_element_assets(self.workspace, element['id'], [upload('one.png'), upload('two.png')])
        self.assertEqual(service.get_element(self.workspace, element['id'])['current_version'], 1)
        self.assertEqual(set(service.ASSET_DIR.rglob('*.png')), before)

    def test_repeated_create_and_add_request_save_once(self):
        first = self.create(request_id='create-request')
        second = self.create(request_id='create-request')
        self.assertEqual(first['id'], second['id'])
        added = service.add_element_assets(self.workspace, first['id'], [upload('profile.png', role='profile')], request_id='add-request')
        again = service.add_element_assets(self.workspace, first['id'], [upload('profile.png', role='profile')], request_id='add-request')
        self.assertEqual(added['current_version_id'], again['current_version_id'])
        self.assertEqual(len(again['assets']), 2)
        with self.assertRaisesRegex(service.ElementError, 'different images'):
            self.create([upload(color='blue')], request_id='create-request')

    def test_phone_orientation_is_normalized_for_preview_and_inference(self):
        output = io.BytesIO()
        image = Image.new('RGB', (320, 256), 'red')
        exif = Image.Exif()
        exif[274] = 6
        image.save(output, format='JPEG', exif=exif)
        saved = self.create([service.UploadedElementAsset('phone.jpg', 'application/octet-stream', output.getvalue())])
        asset = saved['assets'][0]
        self.assertEqual((asset['width'], asset['height']), (256, 320))
        path, mime = service.resolve_element_asset(self.workspace, asset['id'])
        with Image.open(path) as normalized:
            self.assertEqual(normalized.size, (256, 320))
            self.assertNotIn(274, normalized.getexif())
        self.assertEqual(mime, 'image/png')

    def test_face_reference_becomes_canonical_regardless_of_upload_order(self):
        saved = self.create([upload('body.png', role='full_body'), upload('face.png', role='face')])
        self.assertEqual(saved['primary_asset_id'], saved['assets'][1]['id'])

    def test_missing_saved_version_cannot_silently_change_identity(self):
        element = self.create()
        with self.assertRaisesRegex(service.ElementError, "version is unavailable"):
            service.get_element(self.workspace, element['id'], version_id='ev_missing')

    def test_archived_character_cannot_be_rendered_using_an_old_version(self):
        element = self.create()
        service.archive_element(self.workspace, element['id'])
        with self.assertRaisesRegex(service.ElementError, 'archived'):
            service.resolve_element_bindings(self.workspace, [ElementBinding(
                element_id=element['id'], version_id=element['current_version_id'], handle='Actor')])

    def test_labeled_face_closeup_keeps_lower_face_in_reference_sheet(self):
        image = Image.new('RGB', (256, 320), 'red')
        image.paste('blue', (0, 260, 256, 320))
        output = io.BytesIO()
        image.save(output, format='PNG')
        element = self.create([service.UploadedElementAsset('face.png', 'image/png', output.getvalue(), 'face')])
        bindings = service.resolve_element_bindings(self.workspace, [ElementBinding(element_id=element['id'], handle='Actor')])
        sheet = service.build_reference_sheet(bindings, Path(self.temp.name) / 'sheet.png')
        with Image.open(sheet) as reference:
            self.assertIn((0, 0, 255), reference.getdata())
        self.assertEqual(service.elements_for_scene('Actor turns toward the window.', bindings), bindings)

    def test_unbound_handle_fails_before_a_new_identity_can_be_invented(self):
        with self.assertRaisesRegex(service.ElementError, '@Missing'):
            service.validate_prompt_bindings('@Missing walks through the scene.', [])
        service.validate_prompt_bindings('Send to contact@example.com', [])

    def test_actual_multipart_save_retry_and_image_download(self):
        app = FastAPI()
        app.include_router(routes.router, prefix='/elements')
        with patch.object(routes, 'ensure_workspace', return_value=self.workspace), TestClient(app) as client:
            form = dict(name='Actor', handle='Actor', type='character', request_id='network-retry', roles='["face"]')
            files = [('files', ('face.png', upload().data, 'image/png'))]
            first = client.post('/elements', data=form, files=files)
            second = client.post('/elements', data=form, files=files)
            self.assertEqual(first.status_code, 200, first.text)
            self.assertEqual(first.json()['id'], second.json()['id'])
            asset_url = first.json()['assets'][0]['asset_url'].replace('/api/v1', '')
            image = client.get(asset_url)
            self.assertEqual(image.status_code, 200)
            Image.open(io.BytesIO(image.content)).load()
            failed = client.post('/elements/' + first.json()['id'] + '/assets', files=[('files', ('broken.png', b'invalid', 'image/png'))])
            self.assertEqual(failed.status_code, 400)
            self.assertIn('broken.png', failed.json()['detail'])
