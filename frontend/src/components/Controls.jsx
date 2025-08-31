// components/Controls.js
import React from 'react';

const Controls = ({
  isSessionActive,
  onStartSession,
  onStopSession,
  onGenerateVoice,
  isGeneratingVoice,
  selectedLanguage,
  onLanguageChange,
  translatedText
}) => {
  const languages = [
    { code: 'en', name: 'English' },
    { code: 'es', name: 'Spanish' },
    { code: 'fr', name: 'French' },
    { code: 'de', name: 'German' },
    { code: 'it', name: 'Italian' },
    { code: 'pt', name: 'Portuguese' },
    { code: 'ja', name: 'Japanese' },
    { code: 'ko', name: 'Korean' },
    { code: 'zh', name: 'Chinese' },
    { code: 'hi', name: 'Hindi' }
  ];

  return (
    <section className="controls-section">
      <div className="controls-container">
        <div className="session-controls">
          <button
            className={`btn btn-start ${isSessionActive ? 'hidden' : ''}`}
            onClick={onStartSession}
            disabled={isSessionActive}
          >
            <span className="icon">▶</span>
            Start Session
          </button>
          <button
            className={`btn btn-stop ${isSessionActive ? '' : 'hidden'}`}
            onClick={onStopSession}
            disabled={!isSessionActive}
          >
            <span className="icon">⏹</span>
            Stop Session
          </button>
        </div>
        
        <div className="language-selector">
          <label htmlFor="language">Translate to:</label>
          <select
            id="language"
            value={selectedLanguage}
            onChange={onLanguageChange}
            disabled={!isSessionActive}
          >
            {languages.map(lang => (
              <option key={lang.code} value={lang.code}>
                {lang.name}
              </option>
            ))}
          </select>
        </div>
        
        <div className="voice-controls">
          <button
            className={`btn btn-voice ${translatedText ? '' : 'disabled'}`}
            onClick={onGenerateVoice}
            disabled={!translatedText || isGeneratingVoice}
          >
            <span className="icon">
              {isGeneratingVoice ? '⏳' : '🔊'}
            </span>
            {isGeneratingVoice ? 'Generating...' : 'Generate Voice'}
          </button>
        </div>
      </div>
    </section>
  );
};

export default Controls;