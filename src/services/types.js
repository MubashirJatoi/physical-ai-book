// Type definitions for API requests and responses
// Matching the backend contract defined in the OpenAPI specification

// Request types
export const ChatRequestType = {
  query: String,           // The question to ask about the textbook
  session_id: String,      // Unique identifier for the chat session
  user_id: String          // Optional identifier for the user
};

export const SelectedTextChatRequestType = {
  selected_text: String,   // The text selected/highlighted by the user
  query: String,           // The question about the selected text
  session_id: String,      // Unique identifier for the chat session
  user_id: String          // Optional identifier for the user
};

// Response types
export const SourceType = {
  chapter: String,         // The chapter where the information was found
  section: String          // The section where the information was found
};

export const ChatResponseType = {
  answer: String,          // The answer to the user's query
  sources: Array,          // List of sources used to generate the answer
  response_time_ms: Number // Time taken to process the request in milliseconds
};

// Additional types
export const ErrorResponse = {
  error: String            // Error message
};

export const HealthResponse = {
  status: String,          // Overall health status
  services: Object,        // Health status of individual services
  timestamp: String        // Timestamp of the health check
};

// For JavaScript, we also export example structures that can be used for validation
export const exampleChatRequest = {
  query: "Explain ROS 2 nodes and topics",
  session_id: "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  user_id: "user123"
};

export const exampleSelectedTextChatRequest = {
  selected_text: "ROS 2 nodes are a basic unit of composition in the ROS graph.",
  query: "What does this mean in simple terms?",
  session_id: "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  user_id: "user123"
};

export const exampleChatResponse = {
  answer: "ROS 2 nodes are software processes that perform computation...",
  sources: [
    { chapter: "Chapter 1", section: "ROS 2 Architecture" },
    { chapter: "Chapter 2", section: "Node Communication" }
  ],
  response_time_ms: 1250
};