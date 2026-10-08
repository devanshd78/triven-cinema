/** Keep every studio preview in one playback group, including collapsed panels. */
export function createPlaybackCoordinator() {
  const players = new Set<HTMLMediaElement>();

  function pauseAll() {
    for (const player of players) player.pause();
  }

  function register(player: HTMLMediaElement) {
    players.add(player);
    const ownerDocument = player.ownerDocument;
    const ownerWindow = ownerDocument.defaultView;
    const panels: HTMLDetailsElement[] = [];
    let panel = player.parentElement?.closest("details");
    while (panel) {
      panels.push(panel);
      panel = panel.parentElement?.closest("details");
    }

    function onPlay() {
      // A queued play/playing event can arrive after a different player paused it.
      if (player.paused) return;
      if (ownerDocument.hidden || !player.isConnected || player.closest("details:not([open]), [hidden], [inert]")) {
        player.pause();
        return;
      }
      for (const other of players) {
        if (other !== player) other.pause();
      }
    }

    function onToggle() {
      if (panels.some((parent) => !parent.open)) player.pause();
    }

    function onVisibilityChange() {
      if (ownerDocument.hidden) player.pause();
    }

    function onPageHide() {
      player.pause();
    }

    player.addEventListener("play", onPlay);
    player.addEventListener("playing", onPlay);
    for (const parent of panels) parent.addEventListener("toggle", onToggle);
    ownerDocument.addEventListener("visibilitychange", onVisibilityChange);
    ownerWindow?.addEventListener("pagehide", onPageHide);
    onPlay();

    return () => {
      // React also calls this when a preview is removed or its source changes.
      player.pause();
      players.delete(player);
      player.removeEventListener("play", onPlay);
      player.removeEventListener("playing", onPlay);
      for (const parent of panels) parent.removeEventListener("toggle", onToggle);
      ownerDocument.removeEventListener("visibilitychange", onVisibilityChange);
      ownerWindow?.removeEventListener("pagehide", onPageHide);
    };
  }

  return { register, pauseAll };
}

export const studioPlayback = createPlaybackCoordinator();
