/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        void: "#0A0D12",
        surface: "#12171F",
        "surface-raised": "#1A212C",
        line: "#232B38",
        ink: {
          primary: "#E8ECF1",
          muted: "#7C8797",
          faint: "#4A5568",
        },
        // Nucleotide palette -- the design's real accent system. Each color
        // does semantic work (base identity) throughout the UI, not just
        // decoration, so these are named after what they represent rather
        // than generic "primary/secondary" tokens.
        base: {
          // Okabe-Ito colorblind-safe palette (distinguishable under
          // deuteranopia/protanopia/tritanopia). T deliberately sits in
          // reddish-purple rather than pure red, specifically so it can't
          // collapse toward G's green under red-green colorblindness --
          // the most common failure mode a "pick 4 different-looking
          // colors by eye" approach runs into.
          A: "#E69F00", // orange
          T: "#CC79A7", // reddish-purple (was coral-red)
          G: "#009E73", // bluish-green (was pure green)
          C: "#0072B2", // blue
          N: "#6B7280",
        },
        // Deliberately a different hue family from the nucleotide palette above,
        // so heatmap intensity is never visually confused with base identity.
        heat: "#C77DFF",

        // shadcn/ui-convention semantic tokens, sourced from CSS variables
        // (see src/index.css :root) so hand-built Radix-based components
        // (button, tooltip, tabs, dropdown-menu, sonner) share one token
        // system with the rest of the app instead of a second, parallel one.
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        card: { DEFAULT: "hsl(var(--card))", foreground: "hsl(var(--foreground))" },
        popover: { DEFAULT: "hsl(var(--popover))", foreground: "hsl(var(--foreground))" },
        primary: { DEFAULT: "hsl(var(--primary))", foreground: "hsl(var(--primary-foreground))" },
        secondary: { DEFAULT: "hsl(var(--secondary))", foreground: "hsl(var(--foreground))" },
        muted: { DEFAULT: "hsl(var(--muted))", foreground: "hsl(var(--muted-foreground))" },
        accent: { DEFAULT: "hsl(var(--accent))", foreground: "hsl(var(--foreground))" },
        destructive: { DEFAULT: "hsl(var(--destructive))", foreground: "hsl(var(--destructive-foreground))" },
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
      },
      fontFamily: {
        sans: ["IBM Plex Sans", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      boxShadow: {
        panel: "0 1px 0 0 rgba(255,255,255,0.02) inset, 0 8px 24px -12px rgba(0,0,0,0.6)",
      },
      keyframes: {
        "fade-in": { from: { opacity: "0" }, to: { opacity: "1" } },
        "fade-out": { from: { opacity: "1" }, to: { opacity: "0" } },
        "zoom-in": { from: { opacity: "0", transform: "scale(0.96)" }, to: { opacity: "1", transform: "scale(1)" } },
        "zoom-out": { from: { opacity: "1", transform: "scale(1)" }, to: { opacity: "0", transform: "scale(0.96)" } },
        "slide-in-from-top": { from: { transform: "translateY(-4px)", opacity: "0" }, to: { transform: "translateY(0)", opacity: "1" } },
        "slide-in-from-bottom": { from: { transform: "translateY(4px)", opacity: "0" }, to: { transform: "translateY(0)", opacity: "1" } },
      },
      animation: {
        "fade-in": "fade-in 150ms ease-out",
        "fade-out": "fade-out 100ms ease-in",
        "zoom-in": "zoom-in 150ms ease-out",
        "zoom-out": "zoom-out 100ms ease-in",
        "slide-in-from-top": "slide-in-from-top 150ms ease-out",
        "slide-in-from-bottom": "slide-in-from-bottom 150ms ease-out",
      },
    },
  },
  plugins: [],
};
