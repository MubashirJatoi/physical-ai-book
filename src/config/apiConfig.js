// API Configuration for Docusaurus
// Since Docusaurus handles environment variables differently, we define the backend URL here

// For development, use the environment variable if available, otherwise default
// In Docusaurus, you can set window.env in the client modules if needed
const getBackendBaseUrl = () => {
  // Check if we're in a browser environment
  if (typeof window !== 'undefined' && window.env && window.env.REACT_APP_BACKEND_URL) {
    return window.env.REACT_APP_BACKEND_URL;
  }

  // For development, we can use a default value
  // This can be overridden in the Docusaurus config
  return 'http://localhost:8001'; // Updated to match your current backend port
};

export const BACKEND_BASE_URL = getBackendBaseUrl();

// API Endpoints
export const API_ENDPOINTS = {
  CHAT: `${BACKEND_BASE_URL}/chat`,
  CHAT_SELECTED: `${BACKEND_BASE_URL}/chat/selected`,
  INGEST: `${BACKEND_BASE_URL}/ingest`,
  HEALTH: `${BACKEND_BASE_URL}/health`,
  STATS: `${BACKEND_BASE_URL}/stats`,
};

// Default headers for API requests
export const API_HEADERS = {
  'Content-Type': 'application/json',
};