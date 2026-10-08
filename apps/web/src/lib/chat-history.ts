/** Prefer newer local work after a interrupted save, while respecting server deletions. */
export function reconcileChatSessions<T extends { id: string; updatedAt: number; dirty?: boolean }>(remote: T[], local: T[]): T[] {
  const byId = new Map(remote.map((session) => [session.id, session]));
  for (const session of local) {
    const saved = byId.get(session.id);
    if ((!saved && session.dirty !== false) || (saved && session.updatedAt > saved.updatedAt)) byId.set(session.id, session);
  }
  return [...byId.values()].sort((a, b) => b.updatedAt - a.updatedAt);
}

/** A changed scene invalidates its downstream anchors, but preserves earlier work. */
export function invalidateSceneChain<T>(videos: Record<number, T>, sceneIds: number[], changedId: number): Record<number, T> {
  const index = sceneIds.indexOf(changedId);
  if (index < 0) return { ...videos };
  const keep = new Set(sceneIds.slice(0, index));
  return Object.fromEntries(Object.entries(videos).filter(([id]) => keep.has(Number(id))));
}
