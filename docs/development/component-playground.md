# Component playground (dev only)

The component playground lets you view and trigger UI components in isolation during local development, without going through the full flows (e.g. see the report-bug truck animation without submitting the form, or work on the member card drawer without doing a ZIP search).

---

## URL and availability

- **URL:** `GET /dev/playground` with optional `?scene=<id>`, or `GET /dev/playground/<scene_id>` for deep links.
- **Availability:** Only when `ILGA_DEV_MODE=1` (default in dev profile). When `DEV_MODE` is false, these routes return 404.
- **API key:** `/dev` is exempt from API key middleware so the playground loads without headers.
- **Secret shortcut (dev only):** From any page, type the four characters `*dev` (with focus *not* in an input, textarea, or contenteditable). The playground opens in a new tab. The sequence resets after 2 seconds or if you press a wrong key, and does not interfere with normal typing in form fields.

---

## Built-in scenes

| Scene id       | Label                 | Description |
|----------------|-----------------------|-------------|
| `truck`        | Truck animation       | Report-bug truck + status animation; "Run truck animation" button triggers it. |
| `drawer-call`  | Drawer (call view)    | Advocacy drawer shell open with call script partial and mock legislator context. |
| `drawer-email` | Drawer (email view)   | Same drawer shell with email compose partial and mock context. At viewport ≤480px the email drawer is responsive: step counter right-aligned, call banners hidden, From/To one line with ellipsis, subject "Looks good" icon-only, action buttons stacked. |

---

## How to add a new scene

1. **Register the scene** in `src/ilga_graph/dev_playground_scenes.py`:
   - In `get_scenes()`, append a dict with: `id`, `label`, `template` (Jinja template name), `context` (dict or callable returning a dict), and optionally `trigger` (e.g. `"truck"` for a Run button).
   - If the component needs mock data, add a small helper (e.g. `_drawer_call_context()`) and use it as the scene’s `context` (or a callable).

2. **Template:** Either reuse an existing partial (e.g. `_advocacy_drawer_call.html`) or add a new one. For a wrapper that includes a partial:
   - Create a template (e.g. `_dev_playground_drawer_call.html`) that renders the shell (overlay, panel, body container) and `{% include "partial.html" %}` inside the body. The scene’s `context` is passed when the playground renders that template.

3. **Trigger (optional):** If the component needs a "Run" or "Show" action (e.g. truck animation), set `trigger` on the scene and implement the behavior in the scene template (e.g. a button and a small inline script that toggles classes or visibility). The main playground page does not need to know the trigger; the scene template owns it.

4. **Request:** The router always injects `request` into the scene context so templates can use it if needed.

---

## Implementation notes

- **Scene registry:** `dev_playground_scenes.get_scenes()` returns the list; `get_scene(id)` and `get_scene_context(scene, request)` are used by the dev router to resolve template and context.
- **Router:** `src/ilga_graph/routers/dev.py` defines `GET /dev/playground` and `GET /dev/playground/{scene_id}`; both return 404 when not `DEV_MODE`.
- **Playground template:** `dev_playground.html` extends `base.html` (so all site CSS/JS load), shows a nav of scene links, and a main area that either shows "Pick a component" or the selected scene’s rendered HTML (`scene_html`).
- **Truck partial:** The report-bug truck block lives in `_report_bug_truck.html` and is included by both `report_bug.html` and the playground truck scene template (`dev_playground/_scene_truck.html`).
