# Data Model for Physical AI & Humanoid Robotics Textbook

## Content Entities

### Chapter
- **Fields**:
  - id: Unique identifier for the chapter
  - title: Chapter title
  - module: Module number (1-4) the chapter belongs to
  - week: Week number in the 13-week course
  - learningObjectives: Array of 3-5 learning objectives
  - prerequisites: Array of prerequisite knowledge
  - content: Main content in markdown format
  - codeExamples: Array of code examples with explanations
  - exercises: Array of 3-5 hands-on exercises
  - summary: Chapter summary and key takeaways
  - furtherReading: Array of further reading resources
  - previousChapterId: Reference to previous chapter (optional)
  - nextChapterId: Reference to next chapter (optional)
  - createdAt: Creation timestamp
  - updatedAt: Last update timestamp

### Module
- **Fields**:
  - id: Unique identifier for the module
  - title: Module title
  - number: Module number (1-4)
  - weekRange: Range of weeks the module covers (e.g., "Weeks 3-5")
  - description: Module description
  - chapters: Array of chapter IDs in this module
  - createdAt: Creation timestamp
  - updatedAt: Last update timestamp

### Content
- **Fields**:
  - id: Unique identifier for content item
  - type: Type of content (e.g., "text", "code", "image", "video", "exercise")
  - title: Content title
  - body: Content body in appropriate format (markdown, code, etc.)
  - metadata: Additional metadata (alt text for images, duration for videos, etc.)
  - createdAt: Creation timestamp
  - updatedAt: Last update timestamp

## User Entities

### Student
- **Fields**:
  - id: Unique identifier for student
  - name: Student's name
  - email: Student's email address
  - progress: Object tracking progress through chapters
  - preferences: Personalization preferences
  - language: Preferred language (default: "en", options: ["en", "ur"])
  - createdAt: Registration timestamp
  - updatedAt: Last update timestamp

### Educator
- **Fields**:
  - id: Unique identifier for educator
  - name: Educator's name
  - email: Educator's email address
  - institution: Educator's institution
  - courses: Array of courses the educator manages
  - createdAt: Registration timestamp
  - updatedAt: Last update timestamp

## Interactive Feature Entities

### ProgressTracking
- **Fields**:
  - id: Unique identifier for progress record
  - studentId: Reference to student
  - chapterId: Reference to chapter
  - completed: Boolean indicating if chapter is completed
  - progressPercentage: Percentage of chapter completed
  - lastAccessed: Timestamp of last access
  - exercisesCompleted: Array of completed exercise IDs
  - createdAt: Creation timestamp
  - updatedAt: Last update timestamp

### Translation
- **Fields**:
  - id: Unique identifier for translation
  - contentId: Reference to original content
  - language: Target language code (e.g., "ur" for Urdu)
  - translatedContent: Translated content body
  - status: Translation status ("pending", "approved", "rejected")
  - createdAt: Creation timestamp
  - updatedAt: Last update timestamp

## State Transitions

### Student Progress
- **States**: Not Started → In Progress → Completed
- **Transitions**:
  - Not Started → In Progress: When student starts reading a chapter
  - In Progress → Completed: When student completes all exercises and assessments
  - In Progress → Not Started: When student resets progress (optional)

### Content Translation
- **States**: Original → Translation Pending → Translated → Approved
- **Transitions**:
  - Original → Translation Pending: When translation is requested
  - Translation Pending → Translated: When translation is completed
  - Translated → Approved: When translation is reviewed and approved
  - Translated → Translation Pending: When translation needs revision