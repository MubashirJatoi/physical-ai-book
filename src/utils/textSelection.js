// Utility to handle text selection and communication with chatbot
// This file provides functions to set selected text that can be used by the chatbot widget

let selectedTextCallback = null;

// Function to set the selected text callback
export const setSelectedTextCallback = (callback) => {
  selectedTextCallback = callback;
};

// Function to handle text selection on the page
export const handleTextSelection = () => {
  const selectedText = window.getSelection().toString().trim();

  if (selectedText && selectedText.length > 0 && selectedTextCallback) {
    // Limit the selected text to a reasonable length to avoid overwhelming the API
    const limitedText = selectedText.length > 1000 ? selectedText.substring(0, 1000) + '...' : selectedText;
    selectedTextCallback(limitedText);
  }
};

// Initialize text selection listener
export const initTextSelectionListener = () => {
  // Listen for mouse up event to capture text selection
  document.addEventListener('mouseup', handleTextSelection);

  // Also listen for touchend for mobile devices
  document.addEventListener('touchend', handleTextSelection);

  return () => {
    // Cleanup function to remove listeners
    document.removeEventListener('mouseup', handleTextSelection);
    document.removeEventListener('touchend', handleTextSelection);
  };
};

// Function to get currently selected text
export const getSelectedText = () => {
  return window.getSelection().toString().trim();
};