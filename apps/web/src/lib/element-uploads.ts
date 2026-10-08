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
