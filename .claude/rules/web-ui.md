# Web UI Rules

The browser tools (demo, labeling, calibration) are single-file FastAPI apps that serve one
HTML page with inline CSS/JS. Keep them that way — no build step, no framework, no CDN deps.

## Layout & theme

- Dark theme. Use CSS variables, don't hardcode colors repeatedly:
  `--bg:#0e1116; --card:#161b22; --edge:#2b3340; --txt:#e6edf3; --dim:#8b97a5;`
  verdict colors `--green:#3fb950; --amber:#d29922; --red:#f85149`.
- System font stack (`font: 14px system-ui`). No web fonts.
- Video panels: lock aspect ratio with `aspect-ratio:16/9` and `object-fit:contain` so the
  video never stretches or leaves dead black space. Overlay canvas sits `position:absolute;
  inset:0` over the video.
- Split-screen = CSS grid with equal columns (`grid-template-columns:1fr 1fr`). Add side
  panels as extra grid columns only when asked.
- Buttons: subtle, consistent — `#21304b` bg, `#3a4f78` border, rounded. Keep the control bar
  in one row, wrap on small screens.

## Behavior

- The page must not break if a data field is missing — guard every `data.x.y` access and
  every per-frame coordinate (a camera can have no detection for a frame → null, not a crash).
- Browser plays the video natively (`<video>` src to a range-served endpoint). Never push
  decoded frames through Python during playback — that is what makes these UIs lag.
- Overlays update in `requestAnimationFrame`, reading the current `video.currentTime`.
- Two videos stay in sync by nudging the follower's `currentTime`/`playbackRate` toward the
  leader, not by re-seeking every frame.
- Sound via WebAudio oscillators (no audio files): IN = 880 Hz single, OUT = 300 Hz double,
  CLOSE_IN = 1320 Hz double — match `linecall/bounce/sound.py`.

## Real-time intent

- These UIs are the seam to the live system. Structure detection/processing as an async
  worker decoupled from rendering: if processing falls behind, the video keeps playing and
  overlays catch up. Never block the render loop on inference.

## Don't

- No heavy frameworks (React/Vue), no bundlers, no external CSS/JS unless explicitly required.
- No inline comments explaining obvious DOM calls. Keep the `<script>` lean.
