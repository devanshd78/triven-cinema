"use client";

import type { VideoHTMLAttributes } from "react";
import { studioPlayback } from "@/lib/media-playback";

type StudioVideoProps = Omit<VideoHTMLAttributes<HTMLVideoElement>, "autoPlay" | "children" | "src"> & { src: string };

function connectPlayer(player: HTMLVideoElement | null) {
  if (player) return studioPlayback.register(player);
}

export function StudioVideo({ src, ...props }: StudioVideoProps) {
  // A new source gets a fresh media element. The ref cleanup pauses the old one,
  // including pending playback, without changing the saved video or its audio.
  return <video controls playsInline preload="metadata" {...props} key={src} src={src} ref={connectPlayer} />;
}
