# Frontend Configuration Guide

This document explains how to configure the Physical AI textbook frontend for different environments.

## Environment Configuration

The frontend uses a configuration system that supports different environments:

### Development
- By default, the frontend connects to a local backend at `http://localhost:8000`
- To override this, create a `.env` file in the `physical-ai-book` directory:
  ```
  REACT_APP_BACKEND_URL=http://localhost:8000
  ```

### Production
- The frontend is configured to connect to the deployed backend on Hugging Face Spaces
- Update the `BACKEND_URL` in `src/config.js` with your actual Hugging Face Space URL before deployment

## Configuration File

The main configuration is located in `src/config.js` and includes:

- `BACKEND_URL`: The URL of the backend API
- `CHATBOT_CONFIG`: Settings for the chatbot functionality
- `SESSION_CONFIG`: Settings for session management
- `ERROR_CONFIG`: Settings for error handling and retries

## Deployment Configuration

When deploying to production:

1. Update the `BACKEND_URL` in `src/config.js` with your Hugging Face Space URL
2. Build the project: `npm run build`
3. The built files will be in the `build/` directory

## Environment Variables

The following environment variables can be used to customize the frontend:

- `REACT_APP_BACKEND_URL`: Override the backend URL (useful for development)

## Session Management

The frontend maintains chat sessions in the browser's localStorage with a 24-hour timeout. Sessions include:
- Conversation history
- Current session ID
- User preferences (if implemented)

## Error Handling

The frontend includes comprehensive error handling:
- Network error detection and retry logic
- Rate limit handling with appropriate delays
- User-friendly error messages
- Graceful degradation when backend is unavailable