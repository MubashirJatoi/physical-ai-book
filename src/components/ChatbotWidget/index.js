import React, { useState, useEffect } from 'react';
import './ChatbotWidget.css';
import { chatFullBook, chatSelectedText } from '../../services/api';
import { addMessageToSession, getConversationHistory } from '../../services/session';

const ChatbotWidget = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState('');

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  // Function to handle full-book chat
  const sendFullBookMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Call the backend API for full-book chat
      const response = await chatFullBook(inputValue);

      const botMessage = {
        id: Date.now() + 1,
        text: response.answer,
        sender: 'bot',
        timestamp: new Date().toISOString(),
        sources: response.sources || [] // Store sources for display
      };

      // Add both user and bot messages to session for context
      addMessageToSession(userMessage);
      addMessageToSession(botMessage);

      setMessages(prev => [...prev, botMessage]);
      setIsLoading(false);
    } catch (error) {
      console.error('Error sending message to backend:', error);

      // Create an error message to show to the user
      const errorMessage = {
        id: Date.now() + 1,
        text: `I'm sorry, but I encountered an error while processing your request. ${error.message || 'Please try again later.'}`,
        sender: 'bot',
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, errorMessage]);
      setIsLoading(false);
    }
  };

  // Function to handle selected-text chat
  const sendSelectedTextMessage = async () => {
    if (!inputValue.trim() || !selectedText.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      text: `Selected text: ${selectedText}\n\nQuestion: ${inputValue}`,
      sender: 'user',
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Call the backend API for selected-text chat
      const response = await chatSelectedText(selectedText, inputValue);

      const botMessage = {
        id: Date.now() + 1,
        text: response.answer,
        sender: 'bot',
        timestamp: new Date().toISOString(),
        sources: response.sources || [] // Store sources for display
      };

      // Add both user and bot messages to session for context
      addMessageToSession(userMessage);
      addMessageToSession(botMessage);

      setMessages(prev => [...prev, botMessage]);
      setIsLoading(false);
    } catch (error) {
      console.error('Error sending selected text message to backend:', error);

      // Create an error message to show to the user
      const errorMessage = {
        id: Date.now() + 1,
        text: `I'm sorry, but I encountered an error while processing your request. ${error.message || 'Please try again later.'}`,
        sender: 'bot',
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, errorMessage]);
      setIsLoading(false);
    }
  };

  // Function to handle sending a message based on whether there's selected text
  const sendMessage = async () => {
    if (selectedText) {
      await sendSelectedTextMessage();
    } else {
      await sendFullBookMessage();
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Function to clear selected text
  const clearSelectedText = () => {
    setSelectedText('');
  };

  return (
    <div className="chatbot-widget">
      {isOpen ? (
        <div className="chatbot-container">
          <div className="chatbot-header">
            <h3>Textbook Assistant</h3>
            <button className="chatbot-close" onClick={toggleChat}>
              ×
            </button>
          </div>
          <div className="chatbot-messages">
            {messages.length === 0 ? (
              <div className="chatbot-welcome">
                <p>Hello! I'm your AI-powered textbook assistant. Ask me anything about the current chapter or related robotics concepts.</p>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`chatbot-message ${message.sender}-message`}
                >
                  <div className="message-text">{message.text}</div>
                  {/* Display sources if they exist in the bot message */}
                  {message.sender === 'bot' && message.sources && message.sources.length > 0 && (
                    <div className="message-sources">
                      <small>Sources: {message.sources.map(source => `${source.chapter} - ${source.section}`).join(', ')}</small>
                    </div>
                  )}
                  <div className="message-timestamp">
                    {new Date(message.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className="chatbot-message bot-message">
                <div className="message-text">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
          </div>
          <div className="chatbot-input">
            {/* Display selected text if available */}
            {selectedText && (
              <div className="selected-text-preview">
                <small><strong>Context:</strong> {selectedText.substring(0, 100)}{selectedText.length > 100 ? '...' : ''}</small>
                <button
                  className="selected-text-button"
                  onClick={clearSelectedText}
                  title="Clear selected text context"
                >
                  Clear
                </button>
              </div>
            )}
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={selectedText ? "Ask a question about the selected text..." : "Ask a question about this chapter..."}
              rows="2"
            />
            <div className="button-group">
              <button
                onClick={sendMessage}
                disabled={isLoading || !inputValue.trim()}
                className="send-button"
              >
                Send
              </button>
            </div>
          </div>
        </div>
      ) : (
        <button className="chatbot-launcher" onClick={toggleChat}>
          💬
        </button>
      )}
    </div>
  );
};

export default ChatbotWidget;