---
id: 1
title: create-physical-ai-textbook-constitution
stage: constitution
date: 2025-12-06
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: main
user: user
command: /sp.constitution
labels: [constitution, physical-ai, textbook, governance]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - none
---

## Prompt

You are creating an AI-native textbook for "Physical AI & Humanoid Robotics" course using Docusaurus. This textbook must follow these governing principles:

PROJECT GOALS:
- Create a comprehensive textbook covering Physical AI, ROS 2, Gazebo, Unity, NVIDIA Isaac, and VLA (Vision-Language-Action)
- Deploy as a Docusaurus site on GitHub Pages
- Include embedded RAG chatbot for answering questions about content
- Support user authentication, personalization, and Urdu translation features

QUALITY STANDARDS:
- All content must be technically accurate and educational
- Each chapter must include: theory, practical examples, code snippets, and exercises
- Content should be suitable for students learning robotics and AI
- Use clear, structured markdown formatting
- Include diagrams, flowcharts, and visual aids where appropriate
- Code examples must be production-ready and well-commented

CONTENT STRUCTURE:
- Follow the 13-week course outline provided
- Organize into 4 main modules with sub-chapters
- Each chapter should take 1-2 weeks of study
- Include assessments and capstone project details

TECHNICAL REQUIREMENTS:
- Built with Docusaurus v3
- Responsive design for all devices
- Fast loading and optimized for GitHub Pages
- Clean, professional UI/UX
- Proper navigation and search functionality

DOCUMENTATION STANDARDS:
- Use American English spelling
- Include code syntax highlighting
- Add references and citations for technical concepts
- Provide "Further Reading" sections
- Include prerequisites at chapter beginnings

DEVELOPMENT GUIDELINES:
- Use modern React components where needed
- Follow best practices for web accessibility (WCAG 2.1)
- Implement SEO optimization
- Use environment variables for sensitive data
- Write clean, maintainable code

INTEGRATION REQUIREMENTS:
- Space for RAG chatbot integration (FastAPI backend)
- Authentication system placeholder (better-auth)
- Content personalization capability
- Translation system for Urdu language
- Database integration points (Neon Postgres, Qdrant)

The textbook should empower students to understand Physical AI principles, master ROS 2, simulate robots, develop with NVIDIA Isaac, and build conversational humanoid robots.

## Response snapshot

Created the project constitution file for the Physical AI & Humanoid Robotics textbook with the following core principles:
1. Educational Excellence - Ensuring technical accuracy and educational value
2. Structured Content Organization - Following the 13-week course outline
3. Technical Quality Standards - Using proper formatting and production-ready code
4. Comprehensive Coverage - Including all required topics (Physical AI, ROS 2, etc.)
5. Modern Web Standards - Using Docusaurus v3 with accessibility and SEO
6. Integration Ready Architecture - Supporting RAG chatbot, auth, and translations

The constitution establishes governance for the project and defines the technical requirements and development workflow.

## Outcome

- ✅ Impact: Created foundational governance document for the textbook project
- 🧪 Tests: none
- 📁 Files: Updated .specify/memory/constitution.md with project principles
- 🔁 Next prompts: Continue with textbook development using these principles
- 🧠 Reflection: Constitution provides clear guidelines for consistent development

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): PASS
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin textbook content development