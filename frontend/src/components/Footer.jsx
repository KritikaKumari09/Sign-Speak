// components/Footer.jsx
const Footer = () => {
  return (
    <footer className="w-full py-6 px-4 mt-12 bg-gray-900 border-t border-gray-800">
      <div className="container mx-auto max-w-6xl text-center text-gray-400 text-sm">
        <p>
          <span className="text-blue-500 font-semibold">SignSpeak</span> – Sign Language to Voice Translator | Murf AI
        </p>
        <p className="mt-2 text-gray-500">© {new Date().getFullYear()} All rights reserved.</p>
      </div>
    </footer>
  )
}

export default Footer
