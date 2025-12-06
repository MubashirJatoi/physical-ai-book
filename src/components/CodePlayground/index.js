import React, { useState } from 'react';
import './CodePlayground.css';

const CodePlayground = ({
  initialCode = '// Write your code here\nconsole.log("Hello, Robotics!");',
  language = 'javascript',
  title = 'Code Playground'
}) => {
  const [code, setCode] = useState(initialCode);
  const [output, setOutput] = useState('');
  const [isRunning, setIsRunning] = useState(false);

  const runCode = () => {
    setIsRunning(true);
    setOutput('Running code...');

    // Simulate code execution with a timeout
    // In a real implementation, this would connect to a code execution service
    setTimeout(() => {
      try {
        // For security reasons, we can't actually execute arbitrary user code in the browser
        // This is just a simulation for demonstration purposes
        let simulatedOutput = '';

        if (code.includes('console.log')) {
          // Extract and display the logged content
          const logMatches = code.match(/console\.log\(['"`](.*?)['"`]\)/g);
          if (logMatches) {
            simulatedOutput = logMatches
              .map(match => {
                const content = match.match(/['"`](.*?)['"`]/);
                return content ? content[1] : match;
              })
              .join('\n');
          } else {
            simulatedOutput = 'Code executed successfully (no output to display)';
          }
        } else {
          simulatedOutput = 'Code executed successfully';
        }

        setOutput(simulatedOutput);
      } catch (error) {
        setOutput(`Error: ${error.message}`);
      } finally {
        setIsRunning(false);
      }
    }, 1000);
  };

  const resetCode = () => {
    setCode(initialCode);
    setOutput('');
  };

  const copyCode = () => {
    navigator.clipboard.writeText(code);
    // In a real app, you might show a temporary notification
  };

  return (
    <div className="code-playground">
      <div className="playground-header">
        <h4>{title}</h4>
        <div className="playground-actions">
          <button
            className="action-btn run-btn"
            onClick={runCode}
            disabled={isRunning}
            title="Run code"
          >
            {isRunning ? '...' : '▶ Run'}
          </button>
          <button
            className="action-btn reset-btn"
            onClick={resetCode}
            title="Reset code"
          >
            ↺ Reset
          </button>
          <button
            className="action-btn copy-btn"
            onClick={copyCode}
            title="Copy code"
          >
            📋 Copy
          </button>
        </div>
      </div>

      <div className="code-editor">
        <textarea
          value={code}
          onChange={(e) => setCode(e.target.value)}
          className="code-input"
          spellCheck="false"
          aria-label="Code editor"
        />
      </div>

      <div className="code-output">
        <div className="output-header">Output:</div>
        <pre className="output-content">
          {output || '// Output will appear here after running the code'}
        </pre>
      </div>
    </div>
  );
};

export default CodePlayground;