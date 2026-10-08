import assert from 'node:assert/strict';
import test from 'node:test';
import { waitForJobResult, ApiError, GenerationJobError, saveChatHistoryItem, deleteChatHistoryItem } from '../src/lib/api/cinema.ts';
import { reconcileChatSessions, invalidateSceneChain } from '../src/lib/chat-history.ts';

const reply = (data, status = 200) => new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } });

test('polling reconnects after transport failures and retains the original job', async () => {
  const originalFetch = globalThis.fetch;
  const requested = [];
  let count = 0;
  globalThis.fetch = async (url) => {
    requested.push(url);
    if (++count === 1) throw new TypeError('Network offline');
    if (count === 2) return reply({ status: 'running', progress: 70 });
    return reply({ status: 'completed', result: { filename: 'saved.mp4' } });
  };
  try {
    let reconnects = 0;
    const result = await waitForJobResult('original-job', undefined, 1, undefined, () => reconnects++);
    assert.equal(result.filename, 'saved.mp4');
    assert.equal(reconnects, 1);
    assert.deepEqual(new Set(requested), new Set(['/api/v1/generations/jobs/original-job']));
  } finally { globalThis.fetch = originalFetch; }
});

test('a failed later stage exposes saved assets and an authentication failure stops polling', async () => {
  const originalFetch = globalThis.fetch;
  try {
    const assets = [{ filename: 'retained.mp4' }];
    globalThis.fetch = async () => reply({ status: 'failed', error: 'Composition failed', assets });
    await assert.rejects(waitForJobResult('failed-job', undefined, 1), (error) => error instanceof GenerationJobError && error.job.assets[0].filename === 'retained.mp4');
    globalThis.fetch = async () => reply({ detail: 'Authentication required' }, 401);
    await assert.rejects(waitForJobResult('private-job', undefined, 1), (error) => error instanceof ApiError && error.status === 401);
  } finally { globalThis.fetch = originalFetch; }
});

test('aborting a polling subscription makes no further request', async () => {
  const originalFetch = globalThis.fetch;
  const controller = new AbortController();
  let count = 0;
  globalThis.fetch = async () => { count++; return reply({ status: 'running' }); };
  try {
    await assert.rejects(waitForJobResult('job', () => controller.abort(), 1, controller.signal), { name: 'AbortError' });
    assert.equal(count, 1);
  } finally { globalThis.fetch = originalFetch; }
});

test('chat writes and delete remain ordered under slow network responses', async () => {
  const originalFetch = globalThis.fetch;
  const calls = [];
  let release;
  globalThis.fetch = async (_url, init) => {
    calls.push(init.method === 'DELETE' ? 'delete' : JSON.parse(init.body).updated_at);
    if (calls.length === 1) await new Promise((resolve) => { release = resolve; });
    return reply({ id: 'chat-example' });
  };
  try {
    const first = saveChatHistoryItem({ id: 'chat-example', title: 'a', created_at: 1, updated_at: 1, workspace: {} });
    const next = saveChatHistoryItem({ id: 'chat-example', title: 'a', created_at: 1, updated_at: 2, workspace: {} });
    const removed = deleteChatHistoryItem('chat-example');
    await new Promise((resolve) => setImmediate(resolve));
    assert.deepEqual(calls, [1]);
    release();
    await Promise.all([first, next, removed]);
    assert.deepEqual(calls, [1, 2, 'delete']);
  } finally { globalThis.fetch = originalFetch; }
});

test('history keeps unsynced completed takes and respects a synchronized deletion', () => {
  const saved = { id: 'chat-one', updatedAt: 1, dirty: false, video: null };
  const local = { ...saved, updatedAt: 2, dirty: true, video: 'retained.mp4' };
  assert.equal(reconcileChatSessions([saved], [local])[0].video, 'retained.mp4');
  assert.deepEqual(reconcileChatSessions([], [saved]), []);
  assert.equal(reconcileChatSessions([], [local])[0].video, 'retained.mp4');
});

test('regenerating a scene drops its cached take and downstream anchors only', () => {
  const videos = { 10: 'first', 20: 'second', 30: 'third' };
  assert.deepEqual(invalidateSceneChain(videos, [10, 20, 30], 20), { 10: 'first' });
  assert.deepEqual(videos, { 10: 'first', 20: 'second', 30: 'third' });
});

test('logout cancels an old account write while its bootstrap is still in flight', async () => {
  const { logoutCinema } = await import('../src/lib/api/cinema.ts');
  const originalFetch = globalThis.fetch;
  const originalWindow = globalThis.window;
  const calls = [];
  let release;
  globalThis.window = { sessionStorage: { getItem: () => null, setItem() {}, removeItem() {} } };
  globalThis.fetch = async (url) => {
    calls.push(url);
    if (url.endsWith('/bootstrap')) await new Promise((resolve) => { release = resolve; });
    return reply({ ok: true });
  };
  try {
    const oldSave = saveChatHistoryItem({ id: 'chat-account-race', title: 'private account A draft', created_at: 1, updated_at: 1, workspace: {} });
    const rejected = assert.rejects(oldSave, { name: 'AbortError' });
    await new Promise((resolve) => setImmediate(resolve));
    await logoutCinema();
    release();
    await rejected;
    assert.equal(calls.filter((url) => url.includes('/chats/')).length, 0);
  } finally { globalThis.fetch = originalFetch; globalThis.window = originalWindow; }
});
