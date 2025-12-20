// Frontend configuration for the Physical AI textbook chatbot
// This file manages environment-specific configurations

// Determine the backend API URL based on environment
const getBackendUrl = () => {
  // For development, you can set REACT_APP_BACKEND_URL in your .env file
  if (process.env.REACT_APP_BACKEND_URL) {
    return process.env.REACT_APP_BACKEND_URL;
  }

  // For production deployment, use the Hugging Face Space URL
  // Replace with your actual Hugging Face Space URL when deploying
  // Example: https://your-username-chatbot.hf.space
  return process.env.NODE_ENV === 'production'
    ? 'https://your-huggingface-space-url.hf.space'  // Update this with your actual deployment URL
    : 'http://localhost:8000';  // Default for local development
};

const config = {
  // Backend API configuration
  BACKEND_URL: getBackendUrl(),

  // Chatbot configuration
  CHATBOT_CONFIG: {
    // Maximum number of messages to keep in conversation context
    MAX_CONTEXT_MESSAGES: 5,

    // Time in milliseconds to wait before showing typing indicator
    TYPING_INDICATOR_DELAY: 1000,

    // Maximum length of selected text to send to backend
    MAX_SELECTED_TEXT_LENGTH: 5000,
  },

  // Session management configuration
  SESSION_CONFIG: {
    // Session timeout in milliseconds (24 hours)
    TIMEOUT: 24 * 60 * 60 * 1000,

    // Key used to store session data in localStorage
    STORAGE_KEY: 'chatbot_session_data',

    // Key used to store session ID in localStorage
    ID_KEY: 'chatbot_session_id',
  },

  // Error handling configuration
  ERROR_CONFIG: {
    // Maximum number of retry attempts for failed API calls
    MAX_RETRY_ATTEMPTS: 3,

    // Initial delay between retry attempts in milliseconds
    INITIAL_RETRY_DELAY: 1000,
  }
};

export default config;