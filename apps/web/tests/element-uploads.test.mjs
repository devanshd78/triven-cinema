import assert from 'node:assert/strict';
import test from 'node:test';
import { selectReferenceFiles } from '../src/lib/element-uploads.ts';

const file = (name, size = 100) => ({ name, size, lastModified: 100, type: 'application/octet-stream' });
test('adding references keeps the previous selection and ignores a repeated file', () => {
  const first = file('front.PNG');
  const next = selectReferenceFiles([first], [first, file('side.jpg')], 8, 15);
  assert.deepEqual(next.files.map((item) => item.name), ['front.PNG', 'side.jpg']);
  assert.equal(next.error, '');
});
test('invalid files are named without discarding valid selections', () => {
  const next = selectReferenceFiles([file('face.png')], [file('photo.heic'), file('large.jpg', 16 * 1024 * 1024), file('body.webp')], 8, 15);
  assert.match(next.error, /photo.heic/);
  assert.match(next.error, /large.jpg/);
  assert.equal(next.files.length, 2);
});
test('reference count is bounded before upload', () => {
  const next = selectReferenceFiles([file('one.png')], [file('two.png'), file('three.png')], 2, 15);
  assert.equal(next.files.length, 2);
  assert.match(next.error, /at most 2/);
});

test('filenames identify face, profile, body and costume regardless of selection order', async () => {
  const { referenceRoles, referenceFileKey } = await import('../src/lib/element-uploads.ts');
  const images = ['side profile.jpeg', 'full body.jpeg', 'face.jpeg', 'costume.jpeg'].map((name) => file(name));
  assert.deepEqual(referenceRoles('character', images), ['profile', 'full_body', 'face', 'costume']);
  assert.deepEqual(referenceRoles('character', images, { [referenceFileKey(images[0])]: 'face' }), ['face', 'full_body', 'face', 'costume']);
});
test('unnamed references fill the roles not already identified by a filename', async () => {
  const { referenceRoles } = await import('../src/lib/element-uploads.ts');
  assert.deepEqual(referenceRoles('character', [file('photo001.jpg'), file('face.jpg')]), ['full_body', 'face']);
  assert.deepEqual(referenceRoles('prop', [file('face.png')]), ['object']);
});
