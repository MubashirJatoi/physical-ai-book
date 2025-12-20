// Session management utility for maintaining conversation context
// Handles session creation, persistence, and management across page navigation

const SESSION_STORAGE_KEY = 'chatbot_session_data';
const SESSION_ID_KEY = 'chatbot_session_id';
const SESSION_TIMEOUT = 24 * 60 * 60 * 1000; // 24 hours in milliseconds

// Create a new session ID (proper UUID format for backend)
export const createSessionId = () => {
  // Generate a proper UUID v4 format
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
};

// Get the current session ID, creating one if needed
export const getSessionId = () => {
  let sessionId = localStorage.getItem(SESSION_ID_KEY);

  if (!sessionId) {
    sessionId = createSessionId();
    localStorage.setItem(SESSION_ID_KEY, sessionId);
  }

  return sessionId;
};

// Get the current session data
export const getSessionData = () => {
  const sessionData = localStorage.getItem(SESSION_STORAGE_KEY);
  if (!sessionData) {
    return {
      sessionId: getSessionId(),
      createdAt: Date.now(),
      lastInteraction: Date.now(),
      messages: [],
      userId: null
    };
  }

  try {
    const parsed = JSON.parse(sessionData);
    // Check if session is expired (older than 24 hours)
    if (Date.now() - parsed.createdAt > SESSION_TIMEOUT) {
      clearSession();
      return getSessionData(); // Recursively get fresh session data
    }
    return parsed;
  } catch (error) {
    console.error('Error parsing session data:', error);
    clearSession();
    return getSessionData(); // Recursively get fresh session data
  }
};

// Update session data
export const updateSessionData = (updates) => {
  const currentData = getSessionData();
  const updatedData = {
    ...currentData,
    ...updates,
    lastInteraction: Date.now()
  };

  localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(updatedData));
  return updatedData;
};

// Add a message to the session
export const addMessageToSession = (message) => {
  const currentData = getSessionData();
  const updatedMessages = [...currentData.messages, message];

  return updateSessionData({
    messages: updatedMessages,
    lastInteraction: Date.now()
  });
};

// Get conversation history for context
export const getConversationHistory = (maxMessages = 5) => {
  const currentData = getSessionData();
  // Return the most recent messages up to maxMessages
  return currentData.messages.slice(-maxMessages);
};

// Set user ID for the session
export const setUserId = (userId) => {
  return updateSessionData({ userId });
};

// Clear the current session
export const clearSession = () => {
  localStorage.removeItem(SESSION_STORAGE_KEY);
  localStorage.removeItem(SESSION_ID_KEY);
};

// Check if session exists and is valid
export const isSessionValid = () => {
  const currentData = getSessionData();
  return Date.now() - currentData.lastInteraction < SESSION_TIMEOUT;
};

// Get session metadata
export const getSessionMetadata = () => {
  const currentData = getSessionData();
  return {
    sessionId: currentData.sessionId,
    createdAt: currentData.createdAt,
    lastInteraction: currentData.lastInteraction,
    messageCount: currentData.messages.length,
    userId: currentData.userId,
    isValid: isSessionValid()
  };
};

// Export default functions for convenience
export default {
  getSessionId,
  getSessionData,
  updateSessionData,
  addMessageToSession,
  getConversationHistory,
  setUserId,
  clearSession,
  isSessionValid,
  getSessionMetadata
};