# Quickstart Guide for Physical AI & Humanoid Robotics Textbook

## Prerequisites

- Node.js (version 18 or higher)
- npm or yarn package manager
- Git for version control
- A GitHub account for deployment

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/physical-ai-textbook.git
cd physical-ai-textbook
```

### 2. Install Dependencies
```bash
npm install
# OR
yarn install
```

### 3. Start Development Server
```bash
npm run start
# OR
yarn start
```

This will start the development server at `http://localhost:3000` with live reloading.

### 4. Build for Production
```bash
npm run build
# OR
yarn build
```

This generates a static site in the `build/` directory ready for deployment.

## Project Structure Overview

```
physical-ai-textbook/
├── docs/                    # Markdown content for all chapters
│   ├── welcome.md          # Welcome page
│   ├── intro/              # Weeks 1-2 content
│   ├── module1/            # Module 1: The Robotic Nervous System (ROS 2)
│   ├── module2/            # Module 2: The Digital Twin (Gazebo & Unity)
│   ├── module3/            # Module 3: The AI-Robot Brain (NVIDIA Isaac)
│   └── module4/            # Module 4: Vision-Language-Action (VLA)
├── src/                    # Custom React components
│   ├── components/         # Interactive components (chatbot, auth, etc.)
│   ├── pages/              # Custom pages
│   ├── css/                # Custom styles
│   └── theme/              # Docusaurus theme customizations
├── static/                 # Static assets (images, videos, files)
├── docusaurus.config.js    # Docusaurus configuration
├── sidebars.js             # Navigation configuration
└── package.json            # Project dependencies and scripts
```

## Adding New Content

### Creating a New Chapter
1. Create a new markdown file in the appropriate module directory:
   ```bash
   # Example: Adding Chapter 5 under Module 2
   touch docs/module2/chapter5.md
   ```

2. Add frontmatter to your chapter file:
   ```markdown
   ---
   title: Chapter Title
   sidebar_label: Chapter 5
   description: Brief description of the chapter
   keywords: [list, of, relevant, keywords]
   learning_objectives:
     - Objective 1
     - Objective 2
     - Objective 3
   prerequisites:
     - Prerequisite 1
     - Prerequisite 2
   ---

   # Chapter Title

   Content goes here...
   ```

### Adding Interactive Elements
Use custom React components for interactive features:

```markdown
import ChatbotWidget from '@site/src/components/ChatbotWidget';
import ProgressTracker from '@site/src/components/ProgressTracker';
import UrduTranslationToggle from '@site/src/components/UrduTranslationToggle';

# Chapter Title

Learning content...

<ChatbotWidget />
<ProgressTracker />
<UrduTranslationToggle />
```

## Configuration

### Docusaurus Configuration
The `docusaurus.config.js` file contains all site configuration:

- Site metadata (title, tagline, URL)
- Theme configuration
- Plugin settings
- Deployment configuration

### Navigation
The `sidebars.js` file defines the site navigation structure and sidebar organization.

## Deployment to GitHub Pages

### 1. Configure GitHub Pages
In your GitHub repository settings, enable GitHub Pages and select the `gh-pages` branch.

### 2. Set Deployment Configuration
Update the `docusaurus.config.js` file with your GitHub repository details:

```javascript
module.exports = {
  // ...
  organizationName: 'your-github-username',
  projectName: 'physical-ai-textbook',
  deploymentBranch: 'gh-pages',
  // ...
};
```

### 3. Deploy
```bash
npm run deploy
# OR
yarn deploy
```

This command builds the site and pushes the static files to the `gh-pages` branch.

## Development Workflow

### Content Creation
1. Create markdown files in the appropriate module directory
2. Add proper frontmatter with metadata
3. Include learning objectives, prerequisites, exercises, and summaries
4. Use Docusaurus markdown features for enhanced content

### Component Development
1. Create React components in the `src/components/` directory
2. Test components in development mode
3. Ensure components are responsive and accessible

### Testing
- Test locally with `npm run start`
- Verify all links and navigation work correctly
- Check responsive design on different screen sizes
- Validate accessibility compliance