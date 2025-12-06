/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

module.exports = {
  textbookSidebar: [
    {
      type: 'category',
      label: 'Welcome',
      items: [
        'welcome',
        'about',
        'contact',
        'hardware-requirements',
        'assessments'
      ],
    },
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module1/chapter1',
        'module1/chapter2',
        'module1/chapter3',
        'module1/chapter4'
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module2/chapter5',
        'module2/chapter6',
        'module2/chapter7',
        'module2/chapter8'
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)',
      items: [
        'module3/chapter9',
        'module3/chapter10',
        'module3/chapter11',
        'module3/chapter12'
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module4/chapter13',
        'module4/chapter14',
        'module4/chapter15',
        'module4/chapter16',
        'module4/chapter17'
      ],
    },
    {
      type: 'category',
      label: 'Introductory Content',
      items: [
        'intro/week1',
        'intro/week2'
      ],
    }
  ],
};
