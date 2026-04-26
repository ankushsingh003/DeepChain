/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "#0a0a0f",
        bg2: "#11111a",
        bg3: "#18182a",
        border: "rgba(255,255,255,0.08)",
        border2: "rgba(255,255,255,0.14)",
        text: "#f0ede8",
        muted: "rgba(240,237,232,0.5)",
        dim: "rgba(240,237,232,0.28)",
        finance: "#E8D17A",
        "finance-bg": "rgba(232,209,122,0.08)",
        "finance-glow": "rgba(232,209,122,0.18)",
        health: "#6ECFA0",
        "health-bg": "rgba(110,207,160,0.08)",
        "health-glow": "rgba(110,207,160,0.18)",
        legal: "#A78BFA",
        "legal-bg": "rgba(167,139,250,0.08)",
        "legal-glow": "rgba(167,139,250,0.18)",
        mental: "#F4957A",
        "mental-bg": "rgba(244,149,122,0.08)",
        "mental-glow": "rgba(244,149,122,0.18)",
      },
      fontFamily: {
        serif: ['DM Serif Display', 'Georgia', 'serif'],
        sans: ['DM Sans', 'system-ui', 'sans-serif'],
        mono: ['DM Mono', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse-dot 2s ease-in-out infinite',
      },
      keyframes: {
        'pulse-dot': {
          '0%, 100%': { opacity: 1, transform: 'scale(1)' },
          '50%': { opacity: 0.5, transform: 'scale(0.7)' },
        }
      }
    },
  },
  plugins: [],
}
