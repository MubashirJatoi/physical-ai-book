// API Configuration for Docusaurus
// Since Docusaurus handles environment variables differently, we define the backend URL here

// For development, use the environment variable if available, otherwise default
// In production, use the Hugging Face Space URL
const getBackendBaseUrl = () => {
  // Check if we're in a browser environment and have an environment variable
  if (typeof window !== 'undefined' && window.env && window.env.REACT_APP_BACKEND_URL) {
    return window.env.REACT_APP_BACKEND_URL;
  }

  // Check for production environment - if not localhost, use Hugging Face Space
  if (typeof window !== 'undefined' && window.location && window.location.hostname !== 'localhost' && !window.location.hostname.includes('127.0.0.1')) {
    // Production: Use the Hugging Face Space URL
    return 'https://mubashirjatoi-rag-chatbot.hf.space';
  }

  // For development, we can use a default value
  // This can be overridden in the Docusaurus config
  return 'http://localhost:8001'; // Updated to match your current backend port
};

export const BACKEND_BASE_URL = getBackendBaseUrl();

// API Endpoints
export const API_ENDPOINTS = {
  CHAT: `${BACKEND_BASE_URL}/api/chat`,  // Using /api prefix as called in the service
  CHAT_SELECTED: `${BACKEND_BASE_URL}/api/chat/selected`,
  INGEST: `${BACKEND_BASE_URL}/api/ingest`,
  HEALTH: `${BACKEND_BASE_URL}/api/health`,
  STATS: `${BACKEND_BASE_URL}/api/stats`,
};

// Default headers for API requests
export const API_HEADERS = {
  'Content-Type': 'application/json',
};