/** @type {import('tailwindcss').Config} */
module.exports = {
    presets: [require("@spartan-ng/brain/hlm-tailwind-preset")],
    content: ["./src/**/*.{html,ts}", "./src/app/components/ui/**/*.{html,ts}"],
    darkMode: "media",
    theme: {
        extend: {
            fontFamily: {
                inter: ["Inter", "sans-serif"],
            },
            zIndex: {
                inf: 999,
            },
        },
    },
    plugins: [],
};
