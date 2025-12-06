import React from 'react';
import './VideoEmbed.css';

const VideoEmbed = ({ src, title = "Video Content", aspectRatio = "16:9" }) => {
  // Determine if it's a YouTube URL and extract video ID
  const isYouTube = src.includes('youtube.com') || src.includes('youtu.be');
  let embedUrl = src;

  if (isYouTube) {
    let videoId = '';
    if (src.includes('youtu.be')) {
      videoId = src.split('youtu.be/')[1].split(/[?&]/)[0];
    } else {
      videoId = src.split('v=')[1]?.split(/[&?]/)[0];
    }
    if (videoId) {
      embedUrl = `https://www.youtube.com/embed/${videoId}?rel=0&modestbranding=1`;
    }
  }

  // For other video hosting services, you might need to handle differently
  // This is a simplified version for common video embeds

  return (
    <div className="video-embed-container" style={{ aspectRatio }}>
      <div className="video-embed-wrapper">
        <iframe
          src={embedUrl}
          title={title}
          frameBorder="0"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
          className="video-embed-iframe"
        ></iframe>
      </div>
      {title && <div className="video-title">{title}</div>}
    </div>
  );
};

// Alternative component for local video files
const LocalVideoPlayer = ({ src, title = "Video Content", aspectRatio = "16:9" }) => {
  return (
    <div className="video-embed-container" style={{ aspectRatio }}>
      <video
        src={src}
        controls
        className="video-player"
        aria-label={title}
      >
        Your browser does not support the video tag.
      </video>
      {title && <div className="video-title">{title}</div>}
    </div>
  );
};

VideoEmbed.Local = LocalVideoPlayer;
export default VideoEmbed;