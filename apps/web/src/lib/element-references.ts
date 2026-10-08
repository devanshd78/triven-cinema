export function hasElementMention(text: string, handle: string): boolean {
  const escaped = handle.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`(^|[^A-Za-z0-9_])@${escaped}(?![A-Za-z0-9_-])`, "i").test(text);
}

export function appliesToAllScenes(type: string, explicit?: boolean): boolean {
  return explicit ?? (type === "character" || type === "style");
}
