import React, { useState } from 'react';
import './PersonalizationButton.css';

const PersonalizationButton = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [preferences, setPreferences] = useState({
    theme: 'light',
    fontSize: 'medium',
    readingSpeed: 'normal',
    notifications: true
  });

  const toggleMenu = () => {
    setIsOpen(!isOpen);
  };

  const handlePreferenceChange = (key, value) => {
    setPreferences(prev => ({
      ...prev,
      [key]: value
    }));

    // Save preferences to localStorage
    localStorage.setItem('personalizationPreferences', JSON.stringify({
      ...preferences,
      [key]: value
    }));
  };

  const applyPreferences = () => {
    // In a real implementation, this would apply the preferences to the UI
    // For example, changing CSS variables for theme, font size, etc.
    document.documentElement.setAttribute('data-theme', preferences.theme);
    document.documentElement.setAttribute('data-font-size', preferences.fontSize);

    setIsOpen(false);
  };

  return (
    <div className="personalization-container">
      <button
        className="personalization-btn"
        onClick={toggleMenu}
        aria-expanded={isOpen}
        aria-label="Personalization options"
      >
        🎛️ Customize
      </button>

      {isOpen && (
        <div className="personalization-menu">
          <div className="menu-header">
            <h4>Personalization</h4>
            <button className="close-btn" onClick={toggleMenu}>×</button>
          </div>

          <div className="preference-options">
            <div className="preference-group">
              <label>Theme:</label>
              <select
                value={preferences.theme}
                onChange={(e) => handlePreferenceChange('theme', e.target.value)}
              >
                <option value="light">Light</option>
                <option value="dark">Dark</option>
                <option value="auto">Auto (System)</option>
              </select>
            </div>

            <div className="preference-group">
              <label>Font Size:</label>
              <select
                value={preferences.fontSize}
                onChange={(e) => handlePreferenceChange('fontSize', e.target.value)}
              >
                <option value="small">Small</option>
                <option value="medium">Medium</option>
                <option value="large">Large</option>
                <option value="xlarge">Extra Large</option>
              </select>
            </div>

            <div className="preference-group">
              <label>Reading Speed:</label>
              <select
                value={preferences.readingSpeed}
                onChange={(e) => handlePreferenceChange('readingSpeed', e.target.value)}
              >
                <option value="slow">Slow</option>
                <option value="normal">Normal</option>
                <option value="fast">Fast</option>
              </select>
            </div>

            <div className="preference-group checkbox-group">
              <label>
                <input
                  type="checkbox"
                  checked={preferences.notifications}
                  onChange={(e) => handlePreferenceChange('notifications', e.target.checked)}
                />
                Enable notifications
              </label>
            </div>
          </div>

          <div className="menu-actions">
            <button className="apply-btn" onClick={applyPreferences}>
              Apply Settings
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default PersonalizationButton;