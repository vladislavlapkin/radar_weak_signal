import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// В разработке /api проксируется на FastAPI, в Docker то же делает nginx
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://localhost:8000", rewrite: (p) => p.replace(/^\/api/, "") } },
  },
});
