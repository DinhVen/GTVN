import React, { useCallback, useEffect, useRef, useState } from 'react';
import ReactCrop from 'react-image-crop';
import 'react-image-crop/dist/ReactCrop.css';
import './App.css';

const API_BASE_URL = 'http://localhost:8000';
const DEFAULT_CROP = { unit: '%', width: 80, height: 80, x: 10, y: 10 };
const LOADED_IMAGE_CROP = { unit: '%', width: 90, height: 90, x: 5, y: 5 };
const LABEL_ALIASES = {
  'P.127_50': 'P.127',
};
const SHAPES = [
  { value: 'rect', title: 'Chữ nhật', icon: RectIcon },
  { value: 'circle', title: 'Tròn (biển cấm)', icon: CircleIcon },
  { value: 'triangle', title: 'Tam giác (biển nguy hiểm)', icon: TriangleIcon },
];
function RectIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 32 32" aria-hidden="true">
      <rect x="4" y="6" width="24" height="20" fill="none" stroke="currentColor" strokeWidth="3" rx="2" />
    </svg>
  );
}

function CircleIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 32 32" aria-hidden="true">
      <circle cx="16" cy="16" r="12" fill="none" stroke="currentColor" strokeWidth="3" />
    </svg>
  );
}

function TriangleIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 32 32" aria-hidden="true">
      <polygon points="16,3 2,29 30,29" fill="none" stroke="currentColor" strokeWidth="3" strokeLinejoin="round" />
    </svg>
  );
}

function getCropPixels(crop, image) {
  const isPercent = crop.unit === '%';
  return {
    left: isPercent ? (crop.x / 100) * image.width : crop.x,
    top: isPercent ? (crop.y / 100) * image.height : crop.y,
    width: isPercent ? (crop.width / 100) * image.width : crop.width,
    height: isPercent ? (crop.height / 100) * image.height : crop.height,
  };
}

function applyShapeClip(ctx, shape, width, height) {
  if (shape === 'rect') return false;

  ctx.save();
  ctx.beginPath();
  if (shape === 'circle') {
    ctx.ellipse(width / 2, height / 2, width / 2, height / 2, 0, 0, Math.PI * 2);
  } else if (shape === 'triangle') {
    ctx.moveTo(width / 2, 0);
    ctx.lineTo(0, height);
    ctx.lineTo(width, height);
    ctx.closePath();
  }
  ctx.clip();
  return true;
}

function formatGroup(group) {
  if (!group) return '';

  const lower = group.toLowerCase();
  if (lower.includes('cấm')) return 'Biển Cấm';
  if (lower.includes('hiệu lệnh')) return 'Biển Hiệu Lệnh';
  if (lower.includes('nguy hiểm')) return 'Biển Nguy Hiểm';
  if (lower.includes('chỉ dẫn')) return 'Biển Chỉ Dẫn';
  if (lower.includes('phụ')) return 'Biển Phụ';
  return group;
}

function displayMeaning(meaning) {
  return meaning && meaning !== 'nan' && meaning !== 'Unknown' ? meaning : 'Đang cập nhật...';
}

function displayLabel(label) {
  return LABEL_ALIASES[label] || label;
}

function normalizeResults(rawResults) {
  const seenLabels = new Set();

  return rawResults.reduce((items, item) => {
    const label = displayLabel(item.label);
    if (seenLabels.has(label)) return items;

    seenLabels.add(label);
    items.push({ ...item, label });
    return items;
  }, []);
}

function ShapeOverlay({ crop, image, shape }) {
  if (shape === 'rect' || !image || crop.width <= 0 || crop.height <= 0) return null;

  const { left, top, width, height } = getCropPixels(crop, image);

  return (
    <svg
      className="shape-overlay"
      style={{ left, top, width, height }}
      viewBox={`0 0 ${width} ${height}`}
    >
      <defs>
        <mask id="cropShapeMask">
          <rect width={width} height={height} fill="white" />
          {shape === 'circle' && <ellipse cx={width / 2} cy={height / 2} rx={width / 2} ry={height / 2} fill="black" />}
          {shape === 'triangle' && <polygon points={`${width / 2},0 0,${height} ${width},${height}`} fill="black" />}
        </mask>
      </defs>
      <rect width={width} height={height} fill="rgba(0,0,0,0.55)" mask="url(#cropShapeMask)" />
      {shape === 'circle' && (
        <ellipse
          cx={width / 2}
          cy={height / 2}
          rx={width / 2 - 1}
          ry={height / 2 - 1}
          fill="none"
          stroke="#00e5ff"
          strokeWidth="2"
          strokeDasharray="6,4"
        />
      )}
      {shape === 'triangle' && (
        <polygon
          points={`${width / 2},1 1,${height - 1} ${width - 1},${height - 1}`}
          fill="none"
          stroke="#00e5ff"
          strokeWidth="2"
          strokeDasharray="6,4"
        />
      )}
    </svg>
  );
}

function ShapeControls({ selectedShape, onChange }) {
  return (
    <div className="shape-bar">
      <span className="shape-bar-label">Crop:</span>
      {SHAPES.map(({ value, title, icon }) => (
        <button
          key={value}
          className={`shape-btn-sm ${selectedShape === value ? 'active' : ''}`}
          onClick={() => onChange(value)}
          title={title}
          type="button"
        >
          {React.createElement(icon)}
        </button>
      ))}
    </div>
  );
}

function UploadBox({ fileInputRef, onClick, onFileChange }) {
  return (
    <div className="upload-box" onClick={onClick}>
      <div className="upload-text">Chọn ảnh, kéo thả hoặc Nhấn Ctrl+V để dán ảnh</div>
      <input
        type="file"
        className="file-input"
        ref={fileInputRef}
        onChange={onFileChange}
        accept="image/*"
      />
    </div>
  );
}

function ResultCard({ item }) {
  return (
    <div className="result-card">
      <div className="result-image-wrapper">
        {item.image_path ? (
          <img
            src={`${API_BASE_URL}/${item.image_path}`}
            alt={item.label}
            className="result-image"
            onError={(event) => {
              event.target.onerror = null;
              event.target.src = 'https://via.placeholder.com/200?text=Loi+Anh';
            }}
          />
        ) : (
          <div className="missing-image">-</div>
        )}
      </div>
      <div className="result-info">
        <span className="result-score">Độ tương đồng: {(item.score * 100).toFixed(1)}%</span>
        <span className="result-group"><strong>Loại biển:</strong> {formatGroup(item.group)}</span>
        <span className="result-label"><strong>Ký hiệu (Mã biển):</strong> {item.label}</span>
        <p className="result-meaning">
          <strong>Chức năng / Mô tả:</strong> {displayMeaning(item.meaning)}
        </p>
        <div className="result-advice">
          <strong>Lời khuyên:</strong> {item.advice}
        </div>
      </div>
    </div>
  );
}

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [crop, setCrop] = useState(DEFAULT_CROP);
  const [completedCrop, setCompletedCrop] = useState(null);
  const [shapeMask, setShapeMask] = useState('rect');

  const imgRef = useRef(null);
  const fileInputRef = useRef(null);

  const resetCropState = () => {
    setCrop(DEFAULT_CROP);
    setCompletedCrop(null);
    setShapeMask('rect');
  };

  const resetSession = () => {
    setSelectedFile(null);
    setPreview(null);
    setResults([]);
    setError('');
    resetCropState();
  };

  const processFile = useCallback((file) => {
    setSelectedFile(file);
    setResults([]);
    setError('');

    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result);
      resetCropState();
    };
    reader.readAsDataURL(file);
  }, []);

  useEffect(() => {
    const handlePaste = (event) => {
      const file = event.clipboardData?.files?.[0];
      if (file?.type.startsWith('image/')) {
        processFile(file);
      }
    };

    window.addEventListener('paste', handlePaste);
    return () => window.removeEventListener('paste', handlePaste);
  }, [processFile]);

  const handleBoxClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) processFile(file);
  };

  const getCroppedImgBlob = async () => {
    const image = imgRef.current;
    if (!image || !completedCrop || completedCrop.width <= 0 || completedCrop.height <= 0) {
      return selectedFile;
    }

    const canvas = document.createElement('canvas');
    const scaleX = image.naturalWidth / image.width;
    const scaleY = image.naturalHeight / image.height;
    const cropWidth = completedCrop.width * scaleX;
    const cropHeight = completedCrop.height * scaleY;

    canvas.width = cropWidth;
    canvas.height = cropHeight;

    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, cropWidth, cropHeight);

    const clipped = applyShapeClip(ctx, shapeMask, cropWidth, cropHeight);
    ctx.drawImage(
      image,
      completedCrop.x * scaleX,
      completedCrop.y * scaleY,
      cropWidth,
      cropHeight,
      0,
      0,
      cropWidth,
      cropHeight,
    );

    if (clipped) ctx.restore();

    return new Promise((resolve) => {
      canvas.toBlob((blob) => resolve(blob), 'image/jpeg');
    });
  };

  const handleSearch = async () => {
    if (!preview) return;

    setLoading(true);
    setError('');

    try {
      const finalBlob = await getCroppedImgBlob();
      const formData = new FormData();
      formData.append('file', finalBlob, 'cropped_image.jpg');

      const response = await fetch(`${API_BASE_URL}/search`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Error: ${response.statusText}`);
      }

      const data = await response.json();
      setResults(normalizeResults(data.results || []));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1 className="title">Nhận diện Biển Báo Giao Thông Việt Nam</h1>
        <p className="subtitle">Tìm kiếm bằng phương pháp Embeddings (theo QCVN 41)</p>
      </header>

      <main>
        <div className="upload-section">
          {!preview ? (
            <UploadBox
              fileInputRef={fileInputRef}
              onClick={handleBoxClick}
              onFileChange={handleFileChange}
            />
          ) : (
            <div className="preview-container">
              <p className="crop-hint">
                * Kéo chọn vùng biển báo, chọn hình dạng crop phù hợp để kết quả chính xác hơn!
              </p>
              <div className="crop-stage">
                <ReactCrop
                  crop={crop}
                  onChange={(nextCrop) => setCrop(nextCrop)}
                  onComplete={(nextCrop) => setCompletedCrop(nextCrop)}
                >
                  <img
                    src={preview}
                    ref={imgRef}
                    alt="Upload preview"
                    className="preview-image"
                    onLoad={() => setCrop(LOADED_IMAGE_CROP)}
                  />
                </ReactCrop>
                <ShapeOverlay crop={crop} image={imgRef.current} shape={shapeMask} />
              </div>

              <div className="preview-actions">
                <ShapeControls selectedShape={shapeMask} onChange={setShapeMask} />
                <button className="btn-search btn-primary" onClick={handleSearch} disabled={loading} type="button">
                  {loading ? <span className="loader">Đang tải...</span> : 'Cắt ảnh & Nhận diện'}
                </button>
                <button className="btn-search" onClick={resetSession} disabled={loading} type="button">
                  Làm mới
                </button>
              </div>
            </div>
          )}
        </div>

        {error && <div className="error-message">{error}</div>}

        {results.length > 0 && (
          <section className="results-section">
            <h2 className="results-title">Kết quả tương đồng nhất</h2>
            <div className="cards-container">
              {results.map((item) => (
                <ResultCard key={`${item.rank}-${item.label}`} item={item} />
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
