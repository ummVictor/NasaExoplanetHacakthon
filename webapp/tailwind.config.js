/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        'cosmic-blue': 'rgb(15 23 42)',
        'space-purple': 'rgb(88 28 135)',
        'nebula-pink': 'rgb(219 39 119)',
        'star-white': 'rgb(248 250 252)',
        'void': 'rgb(2 6 23)',
      },
      backgroundImage: {
        'cosmic-gradient': 'linear-gradient(135deg, rgb(2 6 23) 0%, rgb(15 23 42) 100%)',
        'light-gradient': 'linear-gradient(135deg, rgb(248 250 252) 0%, rgb(226 232 240) 100%)',
      }
    },
  },
  plugins: [],
}
