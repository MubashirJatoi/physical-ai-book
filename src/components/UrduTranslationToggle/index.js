import React, { useState, useEffect } from 'react';
import './UrduTranslationToggle.css';

const UrduTranslationToggle = () => {
  const [isUrdu, setIsUrdu] = useState(false);
  const [loading, setLoading] = useState(false);

  // Check user preference from localStorage or browser settings
  useEffect(() => {
    const savedPreference = localStorage.getItem('languagePreference');
    if (savedPreference) {
      setIsUrdu(savedPreference === 'ur');
    } else {
      // Default to English unless browser language is Urdu
      const browserLang = navigator.language || navigator.userLanguage;
      setIsUrdu(browserLang.startsWith('ur'));
    }
  }, []);

  const toggleLanguage = async () => {
    setLoading(true);

    // In a real implementation, this would call an API to get translated content
    // For demo purposes, we'll just toggle the state
    const newLanguage = !isUrdu;
    setIsUrdu(newLanguage);
    localStorage.setItem('languagePreference', newLanguage ? 'ur' : 'en');

    // Simulate API call delay
    setTimeout(() => {
      setLoading(false);

      // Trigger a page refresh or content update in a real app
      // This would involve re-rendering the page with translated content
    }, 500);
  };

  return (
    <div className="translation-toggle">
      <button
        className={`toggle-btn ${isUrdu ? 'urdu-mode' : 'english-mode'} ${loading ? 'loading' : ''}`}
        onClick={toggleLanguage}
        disabled={loading}
        aria-label={isUrdu ? "Switch to English" : "Switch to Urdu"}
      >
        {loading ? (
          <span className="loading-spinner">🔄</span>
        ) : (
          <>
            {isUrdu ? ' ENG ' : ' اردو '}
            <span className="arrow">{isUrdu ? '→' : '←'}</span>
            {isUrdu ? ' اردو ' : ' ENG '}
          </>
        )}
      </button>

      {isUrdu && (
        <div className="translation-note">
          <small>متن کا ترجمہ کیا جا رہا ہے</small>
        </div>
      )}
    </div>
  );
};

export default UrduTranslationToggle;