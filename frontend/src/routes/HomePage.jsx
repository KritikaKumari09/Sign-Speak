// pages/HomePage.jsx
import { motion } from "framer-motion";
import { Languages, Volume2, Camera, History } from "lucide-react";
import RecentTranslations from "../components/RecentTranslations";
import Footer from "../components/Footer";
import { useNavigate } from "react-router-dom";

const HomePage = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-950 to-gray-900 text-gray-100 flex flex-col">
      {/* Hero Section */}
      <header className="flex flex-col items-center justify-center text-center py-20 px-6">
        <motion.h1
          className="text-4xl md:text-6xl pb-4 font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent"
          initial={{ opacity: 0, y: -40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          SignSpeak
        </motion.h1>
        <motion.p
          className="mt-4 text-lg md:text-xl text-gray-400 max-w-2xl"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          Bridging the gap between sign language and spoken communication with
          AI-powered translation.
        </motion.p>

        <motion.button
          onClick={() => {
            navigate("/translate")
          }}
          className="mt-8 px-6 py-3 bg-blue-600 cursor-pointer hover:bg-blue-700 rounded-xl text-white font-semibold shadow-lg flex items-center gap-2"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <Camera className="w-5 h-5" />
          Start Translating
        </motion.button>
      </header>

      {/* Features Section */}
      <section className="py-16 px-6 bg-gray-800/40 backdrop-blur-sm">
        <div className="max-w-6xl mx-auto grid md:grid-cols-3 gap-8 text-center">
          <div className="p-6 bg-gray-900 rounded-2xl shadow-lg hover:shadow-xl transition">
            <Languages className="w-10 h-10 text-blue-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">
              Multi-Language Support
            </h3>
            <p className="text-gray-400 text-sm">
              Translate sign language into different spoken languages instantly.
            </p>
          </div>

          <div className="p-6 bg-gray-900 rounded-2xl shadow-lg hover:shadow-xl transition">
            <Volume2 className="w-10 h-10 text-purple-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">Voice Output</h3>
            <p className="text-gray-400 text-sm">
              Listen to translations with natural voice playback.
            </p>
          </div>

          <div className="p-6 bg-gray-900 rounded-2xl shadow-lg hover:shadow-xl transition">
            <History className="w-10 h-10 text-green-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">Recent Translations</h3>
            <p className="text-gray-400 text-sm">
              Keep track of your past translations for quick access.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default HomePage;
