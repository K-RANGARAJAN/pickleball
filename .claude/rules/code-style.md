# Code Style Rules

## Comments — write FEW

Do not write excessive comments. This is the most important rule here.

- Code should be self-explanatory through clear names. Prefer renaming a variable over
  adding a comment that explains it.
- Comment only the **non-obvious**: a tricky algorithm, a workaround, a unit/convention
  that isn't visible in the type (e.g. "offset is in frames, positive = cam2 later"), or
  *why* something is done a certain way when it would otherwise look wrong.
- Never comment the obvious. No `# loop over frames` above a `for` loop, no `# import numpy`,
  no restating the function name in a comment.
- No commented-out code. Delete it; git remembers.
- One concise docstring per public function/class is enough — say what it does and any
  non-obvious contract, not how. No multi-paragraph essays inside source files.
- If a block needs a long explanation, that explanation belongs in `docs/`, not inline.

Rule of thumb: if removing a comment loses no information, it shouldn't have been there.

## Python

- Python 3.10+. Type hints on public function signatures.
- Names: `snake_case` functions/variables, `PascalCase` classes, `UPPER_SNAKE` constants.
- Pixel vs world space is always explicit in names: `px_x/px_y` vs `world_x/world_y` / `*_ft`.
- Dataclasses for plain data carriers (`Detection`, `CameraPose`, `Bounce2D`).
- Lazy-import heavy/optional deps (torch, ultralytics, cv2) inside the function/method that
  needs them, so importing the module doesn't pull a GPU stack.
- Fail loudly with a clear message on real errors; return an empty/None result on a normal
  miss (e.g. no ball detected) rather than raising.
- Keep functions focused. If a function spans the screen, split it.

## Structure

- Core logic lives in `linecall/`; runnable entry points in `scripts/`.
- A new algorithm goes in the matching `linecall/<area>/` module, not in a script.
- Scripts are thin: parse args, call into `linecall/`, write outputs. No heavy logic in a
  script that belongs in the package.

## Tests

- Every non-trivial module has a test in `tests/`. Validate against known answers
  (synthetic geometry with a known result), not just "it runs".
- Keep the suite green. If you change a module, keep its test passing or update it and say why.
