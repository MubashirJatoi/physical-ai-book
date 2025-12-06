import React, { useState } from 'react';
import './ImageGallery.css';

const ImageGallery = ({ images = [], title = "Image Gallery" }) => {
  const [selectedImageIndex, setSelectedImageIndex] = useState(null);

  const openLightbox = (index) => {
    setSelectedImageIndex(index);
  };

  const closeLightbox = () => {
    setSelectedImageIndex(null);
  };

  const nextImage = () => {
    if (selectedImageIndex !== null) {
      setSelectedImageIndex((prevIndex) =>
        prevIndex === images.length - 1 ? 0 : prevIndex + 1
      );
    }
  };

  const prevImage = () => {
    if (selectedImageIndex !== null) {
      setSelectedImageIndex((prevIndex) =>
        prevIndex === 0 ? images.length - 1 : prevIndex - 1
      );
    }
  };

  const handleKeyDown = (e) => {
    if (selectedImageIndex === null) return;

    if (e.key === 'Escape') {
      closeLightbox();
    } else if (e.key === 'ArrowRight') {
      nextImage();
    } else if (e.key === 'ArrowLeft') {
      prevImage();
    }
  };

  React.useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedImageIndex]);

  return (
    <div className="image-gallery">
      <h3 className="gallery-title">{title}</h3>
      <div className="gallery-grid">
        {images.map((image, index) => (
          <div
            key={index}
            className="gallery-item"
            onClick={() => openLightbox(index)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === 'Enter' && openLightbox(index)}
            aria-label={`View image ${index + 1}`}
          >
            <img
              src={image.src}
              alt={image.alt || `Gallery image ${index + 1}`}
              className="gallery-thumb"
            />
            {image.caption && (
              <div className="gallery-caption">{image.caption}</div>
            )}
          </div>
        ))}
      </div>

      {selectedImageIndex !== null && (
        <div className="lightbox-overlay" onClick={closeLightbox}>
          <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
            <button className="lightbox-close" onClick={closeLightbox} aria-label="Close">
              ×
            </button>
            <button className="lightbox-nav lightbox-prev" onClick={(e) => {
              e.stopPropagation();
              prevImage();
            }} aria-label="Previous image">
              &#8249;
            </button>
            <img
              src={images[selectedImageIndex].src}
              alt={images[selectedImageIndex].alt || `Gallery image ${selectedImageIndex + 1}`}
              className="lightbox-image"
            />
            <button className="lightbox-nav lightbox-next" onClick={(e) => {
              e.stopPropagation();
              nextImage();
            }} aria-label="Next image">
              &#8250;
            </button>
            {images[selectedImageIndex].caption && (
              <div className="lightbox-caption">
                {images[selectedImageIndex].caption}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageGallery;