// components/RecentTranslations.jsx
import { PlayCircle, Languages, FileText } from "lucide-react"

const RecentTranslations = ({ translations }) => {
  return (
     <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-2xl shadow-2xl p-6">
      <h2 className="text-xl font-bold text-blue-400 mb-6 text-center">
        Recent Translations
      </h2>

      {translations.length === 0 ? (
        <p className="text-gray-500 text-center italic">
          No recent translations yet
        </p>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {translations.map((translation) => (
            <div
              key={translation.id}
              className="bg-gray-800 border border-gray-700 rounded-xl p-5 hover:shadow-lg hover:border-gray-600 transition-all duration-300 flex flex-col justify-between"
            >
              {/* Text Content */}
              <div className="flex items-start space-x-3">
                <FileText className="w-5 h-5 text-blue-400 flex-shrink-0 mt-1" />
                <p className="text-gray-200 font-medium leading-relaxed">
                  "{translation.text}"
                </p>
              </div>

              {/* Footer: Language + Play */}
              <div className="flex items-center justify-between mt-4">
                <span className="flex items-center space-x-2 text-sm text-gray-400">
                  <Languages className="w-4 h-4 text-purple-400" />
                  <span className="capitalize">{translation.language}</span>
                </span>

                <button
                  onClick={() => {
                    if (translation.audio) {
                      const audio = new Audio(translation.audio)
                      audio.play()
                    }
                  }}
                  className="text-blue-400 hover:text-blue-500 transition-colors"
                  aria-label="Play translation audio"
                >
                  <PlayCircle className="w-7 h-7" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default RecentTranslations