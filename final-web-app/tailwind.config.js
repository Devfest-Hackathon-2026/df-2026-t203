/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#0b0a08',
        concrete: '#3a352b',
        edge: '#4a4334',
        amber: { glow: '#ffd58a', DEFAULT: '#f2a43a', deep: '#c4631a' },
        cream: '#fff0dc',
        acid: '#ff8a1f',
        brush: '#ff8a1f',
      },
      fontFamily: {
        display: ['Gloock', 'serif'],
        stencil: ['"Saira Stencil One"', 'sans-serif'],
        body: ['Figtree', 'sans-serif'],
        script: ['"Great Vibes"', 'cursive'],
      },
    },
  },
  plugins: [],
}
