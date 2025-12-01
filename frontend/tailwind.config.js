/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        charcoal: {
          900: '#1E1E1E',
          800: '#252526',
          700: '#333333',
          600: '#3E3E42',
        },
        neon: {
          blue: '#00F0FF',
          green: '#00FF9D',
          red: '#FF4D4D',
          yellow: '#FFE600',
        },
      },
    },
  },
  plugins: [],
}
