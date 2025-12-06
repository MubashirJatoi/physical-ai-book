import React, { useState, useEffect } from 'react';
import './ProgressTracker.css';

const ProgressTracker = ({ chapterId, userId = 'demo-user' }) => {
  const [progress, setProgress] = useState({
    completed: false,
    progressPercentage: 0,
    exercisesCompleted: []
  });

  // In a real implementation, this would fetch from an API
  // For demo purposes, we'll simulate the progress state
  useEffect(() => {
    const savedProgress = localStorage.getItem(`progress-${chapterId}-${userId}`);
    if (savedProgress) {
      setProgress(JSON.parse(savedProgress));
    } else {
      // Initialize progress based on chapterId
      const initialProgress = {
        completed: false,
        progressPercentage: 0,
        exercisesCompleted: []
      };
      setProgress(initialProgress);
    }
  }, [chapterId, userId]);

  const updateProgress = (newProgress) => {
    setProgress(newProgress);
    // In a real implementation, this would update the backend
    localStorage.setItem(`progress-${chapterId}-${userId}`, JSON.stringify(newProgress));
  };

  const markAsCompleted = () => {
    const newProgress = {
      ...progress,
      completed: true,
      progressPercentage: 100
    };
    updateProgress(newProgress);
  };

  const markExerciseCompleted = (exerciseId) => {
    if (!progress.exercisesCompleted.includes(exerciseId)) {
      const newProgress = {
        ...progress,
        exercisesCompleted: [...progress.exercisesCompleted, exerciseId],
        progressPercentage: Math.min(100, progress.progressPercentage + 25) // Simplified calculation
      };

      // If all exercises are done, mark chapter as completed
      if (newProgress.exercisesCompleted.length >= 3) { // Assuming 3 exercises per chapter
        newProgress.completed = true;
        newProgress.progressPercentage = 100;
      }

      updateProgress(newProgress);
    }
  };

  return (
    <div className="progress-tracker">
      <div className="progress-header">
        <h4>Learning Progress</h4>
      </div>

      <div className="progress-bar-container">
        <div className="progress-bar">
          <div
            className="progress-fill"
            style={{ width: `${progress.progressPercentage}%` }}
          ></div>
        </div>
        <div className="progress-text">{Math.round(progress.progressPercentage)}% Complete</div>
      </div>

      <div className="progress-status">
        <div className={`completion-status ${progress.completed ? 'completed' : 'in-progress'}`}>
          {progress.completed ? '✓ Completed' : 'In Progress'}
        </div>

        <div className="progress-actions">
          {!progress.completed && (
            <button
              className="mark-complete-btn"
              onClick={markAsCompleted}
              disabled={progress.completed}
            >
              Mark Complete
            </button>
          )}
        </div>
      </div>

      <div className="exercise-tracker">
        <h5>Exercises Completed</h5>
        <div className="exercise-list">
          {[1, 2, 3].map((exNum) => (
            <div key={exNum} className="exercise-item">
              <input
                type="checkbox"
                id={`exercise-${exNum}`}
                checked={progress.exercisesCompleted.includes(`ex${exNum}`)}
                onChange={() => markExerciseCompleted(`ex${exNum}`)}
                disabled={progress.completed}
              />
              <label htmlFor={`exercise-${exNum}`}>
                Exercise {exNum}
              </label>
            </div>
          ))}
        </div>
      </div>

      <div className="progress-summary">
        <p>
          <strong>Chapter Status:</strong> {progress.completed ? 'Completed' : 'In Progress'}<br />
          <strong>Exercises Done:</strong> {progress.exercisesCompleted.length}/3
        </p>
      </div>
    </div>
  );
};

export default ProgressTracker;