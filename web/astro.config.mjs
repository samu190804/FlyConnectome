import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import tailwindcss from '@tailwindcss/vite';

// Sitio estático (Netlify). React solo se hidrata en las islas interactivas (visor 3D y gráficas).
export default defineConfig({
  integrations: [react()],
  vite: { plugins: [tailwindcss()], build: { chunkSizeWarningLimit: 2500 } },
});
