import assert from 'node:assert/strict';
import test from 'node:test';
import { createPlaybackCoordinator } from '../src/lib/media-playback.ts';

class PreviewDocument extends EventTarget {
  hidden = false;
  defaultView = new EventTarget();
  setHidden(hidden) {
    this.hidden = hidden;
    this.dispatchEvent(new Event('visibilitychange'));
  }
}

class PreviewPanel extends EventTarget {
  open = true;
  constructor(parentElement = null) {
    super();
    this.parentElement = parentElement;
  }
  closest() { return this; }
  toggle(open) {
    this.open = open;
    this.dispatchEvent(new Event('toggle'));
  }
}

class PreviewPlayer extends EventTarget {
  paused = true;
  isConnected = true;
  currentTime = 7;
  pauseCount = 0;
  constructor(ownerDocument, parentElement = null) {
    super();
    this.ownerDocument = ownerDocument;
    this.parentElement = parentElement;
  }
  closest() {
    let panel = this.parentElement;
    while (panel) {
      if (!panel.open) return panel;
      panel = panel.parentElement;
    }
    return null;
  }
  play() {
    this.paused = false;
    this.dispatchEvent(new Event('play'));
  }
  pause() { this.paused = true; this.pauseCount++; }
}

function previews() {
  const coordinator = createPlaybackCoordinator();
  const document = new PreviewDocument();
  const first = new PreviewPlayer(document);
  const second = new PreviewPlayer(document);
  coordinator.register(first);
  coordinator.register(second);
  return { coordinator, document, first, second };
}

test('starting a second preview pauses the first without losing its position', () => {
  const { first, second } = previews();
  first.play();
  assert.equal(first.paused, false);
  second.play();
  assert.equal(first.paused, true);
  assert.equal(second.paused, false);
  assert.equal(first.currentTime, 7);
  // Browsers dispatch media events asynchronously; stale events must not steal playback.
  first.dispatchEvent(new Event('playing'));
  first.dispatchEvent(new Event('play'));
  assert.equal(second.paused, false);
});

test('collapsing any enclosing panel pauses its video; reopening does not resume it', () => {
  const coordinator = createPlaybackCoordinator();
  const outer = new PreviewPanel();
  const inner = new PreviewPanel(outer);
  const player = new PreviewPlayer(new PreviewDocument(), inner);
  coordinator.register(player);
  player.play();
  outer.toggle(false);
  assert.equal(player.paused, true);
  outer.toggle(true);
  assert.equal(player.paused, true);
  player.play();
  inner.toggle(false);
  assert.equal(player.paused, true);
});

test('a hidden or detached player cannot start or interrupt a visible preview', () => {
  const { coordinator, document, first, second } = previews();
  const panel = new PreviewPanel();
  panel.open = false;
  const hidden = new PreviewPlayer(document, panel);
  coordinator.register(hidden);
  first.play();
  hidden.play();
  second.isConnected = false;
  second.play();
  assert.equal(hidden.paused, true);
  assert.equal(second.paused, true);
  assert.equal(first.paused, false);
});

test('switching browser tabs pauses playback and returning requires an explicit play', () => {
  const { document, first, second } = previews();
  first.play();
  document.setHidden(true);
  assert.equal(first.paused, true);
  second.play();
  assert.equal(second.paused, true);
  document.setHidden(false);
  assert.equal(first.paused, true);
  assert.equal(second.paused, true);
});

test('removing or replacing a video stops it and releases its event listeners', () => {
  const { coordinator, document, first } = previews();
  const panel = new PreviewPanel();
  const removed = new PreviewPlayer(document, panel);
  const release = coordinator.register(removed);
  removed.play();
  release();
  assert.equal(removed.paused, true);
  const stoppedCount = removed.pauseCount;
  panel.toggle(false);
  document.setHidden(true);
  coordinator.pauseAll();
  document.defaultView.dispatchEvent(new Event('pagehide'));
  assert.equal(removed.pauseCount, stoppedCount);
  document.setHidden(false);
  first.play();
  removed.dispatchEvent(new Event('playing'));
  assert.equal(first.paused, false);
});

test('chat/take resets and leaving the page stop every registered preview', () => {
  const { coordinator, document, first, second } = previews();
  first.play();
  coordinator.pauseAll();
  assert.equal(first.paused, true);
  second.play();
  document.defaultView.dispatchEvent(new Event('pagehide'));
  assert.equal(second.paused, true);
});

test('a player can register again after React development cleanup', () => {
  const coordinator = createPlaybackCoordinator();
  const document = new PreviewDocument();
  const player = new PreviewPlayer(document);
  coordinator.register(player)();
  coordinator.register(player);
  player.play();
  assert.equal(player.paused, false);
  document.setHidden(true);
  assert.equal(player.paused, true);
});
