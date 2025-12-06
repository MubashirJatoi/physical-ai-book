const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

// With JSDoc @type annotations, IDEs can provide config autocompletion
/** @type {import('@docusaurus/types').DocusaurusConfig} */
(module.exports = {
  title: 'Physical AI & Humanoid Robotics Textbook',
  tagline: 'A comprehensive textbook for robotics and AI education',
  url: 'https://MubashirJatoi.github.io',
  baseUrl: '/physical-ai-book/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'MubashirJatoi', // Usually your GitHub org/user name.
  projectName: 'physical-ai-book', // Usually your repo name.
  deploymentBranch: 'gh-pages', // GitHub Pages deployment branch

  presets: [
    [
      '@docusaurus/preset-classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          editUrl: 'https://github.com/your-organization/physical-ai-book/edit/main/',
        },
        blog: false, // Disable blog for textbook
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
        gtag: {
          trackingID: 'G-XXXXXXXXXX',
          anonymizeIP: true,
        },
      }),
    ],
  ],

  themes: [
    // Add search functionality
    [
      '@docusaurus/theme-search-algolia',
      {
        // The application ID provided by Algolia
        appId: 'YOUR_APP_ID',
        // Public API key: it is safe to commit it
        apiKey: 'YOUR_SEARCH_API_KEY',
        indexName: 'physical-ai-book',
        contextualSearch: true,
        searchPagePath: 'search',
      },
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: 'Physical AI Textbook',
        logo: {
          alt: 'Physical AI & Humanoid Robotics Textbook Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'doc',
            docId: 'welcome',
            position: 'left',
            label: 'Textbook',
          },
          {to: '/docs/about', label: 'About', position: 'left'},
          {
            href: 'https://github.com/your-organization/physical-ai-book',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Textbook',
            items: [
              {
                label: 'Welcome',
                to: '/docs/welcome',
              },
              {
                label: 'Hardware Requirements',
                to: '/docs/hardware-requirements',
              },
              {
                label: 'Assessments',
                to: '/docs/assessments',
              },
            ],
          },
          {
            title: 'Modules',
            items: [
              {
                label: 'ROS 2 (Module 1)',
                to: '/docs/module1/chapter1',
              },
              {
                label: 'Simulation (Module 2)',
                to: '/docs/module2/chapter5',
              },
              {
                label: 'AI Integration (Module 3)',
                to: '/docs/module3/chapter9',
              },
              {
                label: 'VLA (Module 4)',
                to: '/docs/module4/chapter13',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/your-organization/physical-ai-book',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook. Built with Docusaurus.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),
});
