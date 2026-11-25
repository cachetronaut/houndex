import { describe, expect, it } from 'vitest';
import { type Embedder, embedderVersion } from './types.js';

const plain: Embedder = {
  dimension: 2,
  embed: async (texts) => texts.map(() => [0, 1]),
};

const named: Embedder = {
  ...plain,
  version: async () => 'model@sha256:abc',
};

describe('embedderVersion', () => {
  it('is undefined when the embedder names no version', async () => {
    expect(await embedderVersion(plain)).toBeUndefined();
  });

  it('returns the version the embedder names', async () => {
    expect(await embedderVersion(named)).toBe('model@sha256:abc');
  });
});
