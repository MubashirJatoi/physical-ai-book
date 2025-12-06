---
sidebar_position: 2
---

import ChatbotWidget from '@site/src/components/ChatbotWidget';
import ProgressTracker from '@site/src/components/ProgressTracker';
import UrduTranslationToggle from '@site/src/components/UrduTranslationToggle';
import PersonalizationButton from '@site/src/components/PersonalizationButton';
import ImageGallery from '@site/src/components/ImageGallery';
import VideoEmbed from '@site/src/components/VideoEmbed';
import CodePlayground from '@site/src/components/CodePlayground';

# Create a Document

<ChatbotWidget />
<ProgressTracker chapterId="tutorial-create-a-document" />
<UrduTranslationToggle />
<PersonalizationButton />

Documents are **groups of pages** connected through:

- a **sidebar**
- **previous/next navigation**
- **versioning**

## Create your first Doc

Create a markdown file at `docs/hello.md`:

```md title="docs/hello.md"
# Hello

This is my **first Docusaurus document**!
```

A new document is now available at `http://localhost:3000/docs/hello`.

## Configure the Sidebar

Docusaurus automatically **creates a sidebar** from the `docs` folder.

Add metadatas to customize the sidebar label and position:

```md title="docs/hello.md" {1-4}
---
sidebar_label: 'Hi!'
sidebar_position: 3
---

# Hello

This is my **first Docusaurus document**!
```

It is also possible to create your sidebar explicitly in `sidebars.js`:

```diff title="sidebars.js"
module.exports = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Tutorial',
-     items: [...],
+     items: ['hello'],
    },
  ],
};
```
