import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Deployed as a GitHub Pages *project* site (mdw22.github.io/weeklyfantasyanalyzer/),
// so every asset URL needs the repo name as a base path.
export default defineConfig({
  base: "/weeklyfantasyanalyzer/",
  plugins: [react()],
});
