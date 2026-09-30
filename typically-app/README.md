# typically-app
Product web app for Typically (Vite + React + TS + Tailwind v4 + shadcn/ui).
Dev: `npm run dev`; needs the API: `uv run --no-sync uvicorn --app-dir site server:app --host 127.0.0.1 --port 8787` (from the repo root).
Build: `npm run build` writes to `../site/app` (gitignored, generated).
Served by the FastAPI server at http://127.0.0.1:8787/app/ (SPA fallback for `/app/*`).
Path alias `@/` -> `src/`; add components with `npx shadcn@latest add <name>`.
