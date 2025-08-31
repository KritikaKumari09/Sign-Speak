// components/TranslationSection.jsx

import { motion, AnimatePresence } from "framer-motion";
import { useState } from "react";
import {
  Globe,
  ChevronDown,
  Loader2,
  Volume2,
  Languages,
  MessageSquare,
} from "lucide-react";

const TranslationSection = ({
  translatedText,
  selectedLanguage,
  languages,
  onGenerateVoice,
  onLanguageChange,
  isAudioGenerating,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  const selected = languages.find((lang) => lang.id === selectedLanguage);

  return (
    <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-2xl shadow-2xl p-6 max-w-[900px] mx-auto border border-gray-700">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
          Translation
        </h2>
        <div className="w-10 h-10 rounded-full bg-blue-500/0 flex items-center justify-center">
          <Languages className="w-6 h-6 text-white" />
        </div>
      </div>

      <div className="relative bg-gray-850 rounded-xl p-5 mb-6 h-56 overflow-y-auto flex items-center justify-center border border-gray-700 shadow-inner">
        {translatedText ? (
          <p className="text-md text-gray-100 leading-relaxed">
            {translatedText}
          </p>
        ) : (
          <div className="text-center text-gray-400 flex items-center">
            {/* <MessageSquare className="w-10 h-10 mb-3 opacity-60 text-white " /> */}
            <p className="text-sm">
              Translated text will appear here once the session starts
            </p>
          </div>
        )}

        {/* Subtle gradient overlay at bottom */}
        <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-gray-850 to-transparent pointer-events-none"></div>
      </div>

      <div className="flex flex-col lg:flex-row gap-4">
        <button
          onClick={onGenerateVoice}
          disabled={!translatedText || isAudioGenerating}
          className={`py-3 px-6 rounded-xl font-medium flex items-center justify-center transition-all duration-300 ${
            !translatedText || isAudioGenerating
              ? "bg-gray-700 text-gray-500 cursor-not-allowed"
              : "bg-gradient-to-r cursor-pointer from-purple-600 to-indigo-600 text-white hover:from-purple-700 hover:to-indigo-700 transform hover:-translate-y-0.5 shadow-lg hover:shadow-purple-500/20"
          }`}
        >
          {isAudioGenerating ? (
            <>
              <Loader2 className="w-5 animate-spin mr-2" /> Generating Audio
            </>
          ) : (
            <>
              <Volume2 className="w-5 h-5 mr-2 text-gray-200" />
              Generate Voice
            </>
          )}
        </button>

        <div className="flex-grow relative">
          {/* Trigger Button */}
          <button
            // onClick={() => setIsOpen((prev) => !prev)}
            className="w-full bg-gray-800 text-gray-100 font-medium py-3 pl-4 pr-10 rounded-xl border border-gray-700 hover:border-gray-600 transition-colors flex items-center justify-between shadow-sm"
          >
            <div className="flex items-center space-x-2">
              <Globe className="w-5 h-5 text-blue-400" />
              <span>
                {selected
                  ? `${selected.flag} ${selected.name}`
                  : "Select Language"}
              </span>
            </div>
            {/* <ChevronDown
              className={`w-5 h-5 cursor-pointer text-blue-400 transition-transform duration-300 ${
                isOpen ? "rotate-180" : "rotate-0"
              }`}
            /> */}
          </button>

          {/* Dropdown Menu */}
          <AnimatePresence>
            {isOpen && (
              <motion.ul
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.2 }}
                className="absolute max-h-[200px] overflow-y-auto z-20 w-full mt-2 bg-gray-800 border border-gray-700 rounded-xl shadow-lg overflow-hidden"
              >
                {languages.map((lang) => (
                  <li
                    key={lang.id}
                    onClick={() => {
                      onLanguageChange({ target: { value: lang.id } });
                      setIsOpen(false);
                    }}
                    className="px-4 py-3 cursor-pointer text-gray-100 hover:bg-gray-700 transition-colors flex items-center space-x-2"
                  >
                    <span>{lang.flag}</span>
                    <span>{lang.name}</span>
                  </li>
                ))}
              </motion.ul>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Status indicator */}
      <div className="mt-5 flex items-center justify-center text-sm text-gray-500">
        <div
          className={`w-2 h-2 rounded-full mr-2 ${
            translatedText ? "bg-green-500" : "bg-gray-600"
          }`}
        ></div>
        <span>
          {translatedText ? "Translation ready" : "Waiting for input"}
        </span>
      </div>
    </div>
  );
};

export default TranslationSection;
