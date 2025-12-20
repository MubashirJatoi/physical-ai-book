// API service for chatbot functionality
// Handles communication with the backend API endpoints

// Import the API configuration
import { BACKEND_BASE_URL } from '../config/apiConfig';

// Create a unique session ID if one doesn't exist (proper UUID format for backend)
const createSessionId = () => {
  const existingSessionId = localStorage.getItem('chatbot_session_id');
  if (existingSessionId) {
    return existingSessionId;
  }

  // Generate a proper UUID v4 format
  const newSessionId = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });

  localStorage.setItem('chatbot_session_id', newSessionId);
  return newSessionId;
};

// Sanitize user input to prevent injection attacks
const sanitizeInput = (input) => {
  if (typeof input !== 'string') return '';

  // Remove potentially dangerous characters/sequences
  return input
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .replace(/javascript:/gi, '')
    .replace(/vbscript:/gi, '')
    .replace(/on\w+=/gi, '')
    .trim();
};

// API call for full-book chat
export const chatFullBook = async (query, sessionId = null, userId = null) => {
  try {
    const session_id = sessionId || createSessionId();
    const sanitizedQuery = sanitizeInput(query);

    if (!sanitizedQuery) {
      throw new Error('Query cannot be empty');
    }

    console.log('Making API call to:', `${BACKEND_BASE_URL}/api/chat`, 'with data:', {
      query: sanitizedQuery,
      session_id,
      user_id: userId || null
    });

    const response = await fetch(`${BACKEND_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: sanitizedQuery,
        session_id,
        user_id: userId || null
      })
    });

    console.log('Response received:', response.status, response.statusText);

    if (!response.ok) {
      const errorText = await response.text().catch(() => 'Unknown error');
      console.error('API call failed with response:', errorText);
      throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
    }

    const data = await response.json();
    console.log('API response data:', data);
    return data;
  } catch (error) {
    console.error('Error in full-book chat:', error);
    // Re-throw with more context about the URL being used
    const fullError = new Error(`Failed to connect to backend at ${BACKEND_BASE_URL}/api/chat. ${error.message}`);
    fullError.originalError = error;
    throw fullError;
  }
};

// API call for selected-text chat
export const chatSelectedText = async (selectedText, query, sessionId = null, userId = null) => {
  try {
    const session_id = sessionId || createSessionId();
    const sanitizedSelectedText = sanitizeInput(selectedText);
    const sanitizedQuery = sanitizeInput(query);

    if (!sanitizedSelectedText) {
      throw new Error('Selected text cannot be empty');
    }

    if (!sanitizedQuery) {
      throw new Error('Query cannot be empty');
    }

    console.log('Making selected-text API call to:', `${BACKEND_BASE_URL}/api/chat/selected`, 'with data:', {
      selected_text: sanitizedSelectedText,
      query: sanitizedQuery,
      session_id,
      user_id: userId || null
    });

    const response = await fetch(`${BACKEND_BASE_URL}/api/chat/selected`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        selected_text: sanitizedSelectedText,
        query: sanitizedQuery,
        session_id,
        user_id: userId || null
      })
    });

    console.log('Response received:', response.status, response.statusText);

    if (!response.ok) {
      const errorText = await response.text().catch(() => 'Unknown error');
      console.error('API call failed with response:', errorText);
      throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
    }

    const data = await response.json();
    console.log('API response data:', data);
    return data;
  } catch (error) {
    console.error('Error in selected-text chat:', error);
    // Re-throw with more context about the URL being used
    const fullError = new Error(`Failed to connect to backend at ${BACKEND_BASE_URL}/api/chat/selected. ${error.message}`);
    fullError.originalError = error;
    throw fullError;
  }
};

// Health check for the backend
export const checkHealth = async () => {
  try {
    console.log('Making health check to:', `${BACKEND_BASE_URL}/api/health`);
    const response = await fetch(`${BACKEND_BASE_URL}/api/health`);
    console.log('Health check response:', response.status, response.statusText);
    if (!response.ok) {
      const errorText = await response.text().catch(() => 'Unknown error');
      console.error('Health check failed with response:', errorText);
      return { status: 'unhealthy', error: `Health check failed with status: ${response.status}, message: ${errorText}` };
    }
    const data = await response.json();
    console.log('Health check data:', data);
    return data;
  } catch (error) {
    console.error('Health check failed:', error);
    return { status: 'unhealthy', error: `Failed to connect to backend health endpoint at ${BACKEND_BASE_URL}/api/health. ${error.message}` };
  }
};

// Get current session ID
export const getCurrentSessionId = () => {
  return localStorage.getItem('chatbot_session_id') || createSessionId();
};

// Clear current session
export const clearSession = () => {
  localStorage.removeItem('chatbot_session_id');
};