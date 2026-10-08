"use client";

import { useEffect, useRef } from "react";

export function ReferenceUploadPreview({ file }: { file: File }) {
  const image = useRef<HTMLImageElement>(null);
  useEffect(() => {
    const url = URL.createObjectURL(file);
    if (image.current) image.current.src = url;
    return () => URL.revokeObjectURL(url);
  }, [file]);
  // The browser previews a local File; it has no server URL for image optimization.
  // eslint-disable-next-line @next/next/no-img-element
  return <img ref={image} alt={`Reference: ${file.name}`} width={56} height={56} />;
}
