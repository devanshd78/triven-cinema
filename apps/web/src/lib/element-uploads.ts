export function referenceFileKey(file: Pick<File, "name" | "size" | "lastModified">): string {
  return `${file.name}:${file.size}:${file.lastModified}`;
}

export function selectReferenceFiles(current: File[], incoming: File[], maxCount: number, maxMb: number) {
  const files = [...current];
  const errors: string[] = [];
  for (const file of incoming) {
    if (files.some((saved) => referenceFileKey(saved) === referenceFileKey(file))) continue;
    if (!/\.(png|jpe?g|webp)$/i.test(file.name) && !["image/png", "image/jpeg", "image/webp"].includes(file.type)) {
      errors.push(`${file.name}: export as PNG, JPEG or WEBP first.`);
    } else if (!file.size || file.size > maxMb * 1024 * 1024) {
      errors.push(`${file.name}: choose a nonempty image smaller than ${maxMb} MB.`);
    } else if (files.length >= maxCount) {
      errors.push(`Choose at most ${maxCount} reference images.`);
    } else {
      files.push(file);
    }
  }
  return { files, error: [...new Set(errors)].join(" ") };
}

export function referenceRoles(type: ElementType, files: File[], overrides: Record<string, ElementAssetRole> = {}, existing: ElementAssetRole[] = []): ElementAssetRole[] {
  const named = files.map((file) => {
    if (overrides[referenceFileKey(file)]) return overrides[referenceFileKey(file)];
    if (type !== "character") return type === "prop" ? "object" : type;
    const name = file.name.replace(/\.[^.]+$/, "").toLowerCase().replace(/[_-]+/g, " ");
    if (/\b(profile|side)\b/.test(name)) return "profile";
    if (/\b(full\s*body|full\s*length|body)\b/.test(name)) return "full_body";
    if (/\b(costume|outfit|wardrobe)\b/.test(name)) return "costume";
    if (/\b(face|portrait|headshot|close\s*up)\b/.test(name)) return "face";
    return undefined;
  });
  const used = new Set([...existing, ...named.filter((role) => role !== undefined)]);
  const available: ElementAssetRole[] = ["face", "full_body", "profile", "costume"];
  return named.map((role) => {
    if (role) return role;
    const next = available.find((value) => !used.has(value)) || "support";
    used.add(next);
    return next;
  });
}
import type { ElementAssetRole, ElementType } from "./types/generation";
