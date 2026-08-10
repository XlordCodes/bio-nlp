import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev-server proxy forwards /api/* to the FastAPI backend (see
// src/api/client.js), so the browser never needs CORS configured for local
// development. In production, set VITE_API_BASE_URL to the deployed
// backend's real origin instead (see .env.example).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
