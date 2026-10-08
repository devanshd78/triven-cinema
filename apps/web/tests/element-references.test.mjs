import assert from 'node:assert/strict';
import test from 'node:test';
import { hasElementMention, appliesToAllScenes } from '../src/lib/element-references.ts';

test('handles match exactly without activating another character with a shared prefix', () => {
  assert.equal(hasElementMention('@Ann-Jane waves.', 'Ann'), false);
  assert.equal(hasElementMention('@ann-jane waves.', 'Ann-Jane'), true);
  assert.equal(hasElementMention('@Radha- waves.', 'Radha-'), true);
  assert.equal(hasElementMention('contact@Ann', 'Ann'), false);
});
test('a typed Character mention carries across scenes unless explicitly scoped', () => {
  assert.equal(appliesToAllScenes('character'), true);
  assert.equal(appliesToAllScenes('character', false), false);
  assert.equal(appliesToAllScenes('prop'), false);
  assert.equal(appliesToAllScenes('prop', true), true);
});
