import React, { useState, useEffect } from 'react';
import './ChatbotWidget.css';

const ChatbotWidget = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  const sendMessage = async () => {
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
      // In a real implementation, this would call the backend API
      // const response = await fetch('/api/chatbot/query', {
      //   method: 'POST',
      //   headers: {
      //     'Content-Type': 'application/json',
      //   },
      //   body: JSON.stringify({
      //     query: inputValue,
      //     userId: 'current-user-id', // Would come from auth context
      //     context: 'current-chapter' // Would come from current page context
      //   })
      // });
      // const data = await response.json();

      // For demo purposes, simulate an API response
      setTimeout(() => {
        const botMessage = {
          id: Date.now() + 1,
          text: `I understand you're asking about "${inputValue}". This is a sample response from the AI-powered chatbot. In a full implementation, this would query the textbook content using RAG (Retrieval Augmented Generation) to provide accurate answers based on the chapter content.`,
          sender: 'bot',
          timestamp: new Date().toISOString()
        };
        setMessages(prev => [...prev, botMessage]);
        setIsLoading(false);
      }, 1000);
    } catch (error) {
      console.error('Error sending message:', error);
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
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
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask a question about this chapter..."
              rows="2"
            />
            <button
              onClick={sendMessage}
              disabled={isLoading || !inputValue.trim()}
              className="send-button"
            >
              Send
            </button>
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