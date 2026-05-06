import React, { useState, useRef, useEffect } from 'react';
import ReactCrop from 'react-image-crop';
import 'react-image-crop/dist/ReactCrop.css';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const [crop, setCrop] = useState({ unit: '%', width: 80, height: 80, x: 10, y: 10 });
  const [completedCrop, setCompletedCrop] = useState(null);
  const [shapeMask, setShapeMask] = useState('rect'); // 'rect' | 'circle' | 'triangle'
  const imgRef = useRef(null);

  const fileInputRef = useRef(null);

  // Add paste functionality
  useEffect(() => {
    const handlePaste = (e) => {
      if (e.clipboardData && e.clipboardData.files && e.clipboardData.files.length > 0) {
        const file = e.clipboardData.files[0];
        if (file.type.startsWith('image/')) {
          processFile(file);
        }
      }
    };
    window.addEventListener('paste', handlePaste);
    return () => window.removeEventListener('paste', handlePaste);
  }, []);

  const handleBoxClick = () => {
    fileInputRef.current.click();
  };

  const processFile = (file) => {
    setSelectedFile(file);
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result);
      // Reset crop state
      setCrop({ unit: '%', width: 80, height: 80, x: 10, y: 10 });
      setCompletedCrop(null);
      setShapeMask('rect');
    };
    reader.readAsDataURL(file);
    setResults([]);
    setError('');
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      processFile(file);
    }
  };

  const getCroppedImgBlob = async () => {
    const image = imgRef.current;
    if (!image || !completedCrop || completedCrop.width <= 0 || completedCrop.height <= 0) {
      return selectedFile;
    }

    const canvas = document.createElement('canvas');
    const scaleX = image.naturalWidth / image.width;
    const scaleY = image.naturalHeight / image.height;

    const cw = completedCrop.width * scaleX;
    const ch = completedCrop.height * scaleY;
    canvas.width = cw;
    canvas.height = ch;

    const ctx = canvas.getContext('2d');

    // Fill white background first (so masked areas are white, not transparent)
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, cw, ch);

    // Apply shape mask
    if (shapeMask === 'circle') {
      ctx.save();
      ctx.beginPath();
      ctx.ellipse(cw / 2, ch / 2, cw / 2, ch / 2, 0, 0, Math.PI * 2);
      ctx.clip();
    } else if (shapeMask === 'triangle') {
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(cw / 2, 0);         // Top center
      ctx.lineTo(0, ch);              // Bottom left
      ctx.lineTo(cw, ch);             // Bottom right
      ctx.closePath();
      ctx.clip();
    }

    ctx.drawImage(
      image,
      completedCrop.x * scaleX,
      completedCrop.y * scaleY,
      cw,
      ch,
      0,
      0,
      cw,
      ch
    );

    if (shapeMask !== 'rect') {
      ctx.restore();
    }

    return new Promise((resolve) => {
      canvas.toBlob((blob) => {
        resolve(blob);
      }, 'image/jpeg');
    });
  };

  const handleSearch = async () => {
    if (!preview) return;

    setLoading(true);
    setError('');

    try {
      const finalBlob = await getCroppedImgBlob();
      const formData = new FormData();
      formData.append('file', finalBlob, "cropped_image.jpg");

      const response = await fetch('http://localhost:8000/search', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Error: ${response.statusText}`);
      }

      const data = await response.json();
      setResults(data.results || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };



  const formatGroup = (group) => {
    if (!group) return "";
    const lower = group.toLowerCase();
    if (lower.includes('cấm')) return "Biển Cấm";
    if (lower.includes('hiệu lệnh')) return "Biển Hiệu Lệnh";
    if (lower.includes('nguy hiểm')) return "Biển Nguy Hiểm";
    if (lower.includes('chỉ dẫn')) return "Biển Chỉ Dẫn";
    return group;
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1 className="title">Nhận diện Biển Báo Giao Thông Việt Nam</h1>
        <p className="subtitle">Tìm kiếm bằng phương pháp Embeddings (theo QCVN 41)</p>
      </header>

      <main>
        <div className="upload-section">
          {!preview ? (
            <div className="upload-box" onClick={handleBoxClick}>
              <div className="upload-text">Chọn ảnh, kéo thả hoặc Nhấn Ctrl+V để dán ảnh</div>
              <input
                type="file"
                className="file-input"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="image/*"
              />
            </div>
          ) : (
            <div className="preview-container">
              <p style={{ marginBottom: '5px', fontSize: '13px', color: 'red' }}>
                * Kéo chọn vùng biển báo, chọn hình dạng crop phù hợp để kết quả chính xác hơn!
              </p>
              <div style={{ position: 'relative', display: 'inline-block' }}>
              <ReactCrop
                crop={crop}
                onChange={c => setCrop(c)}
                onComplete={c => setCompletedCrop(c)}
              >
                <img
                  src={preview}
                  ref={imgRef}
                  alt="Upload preview"
                  className="preview-image"
                  onLoad={() => {
                    setCrop({ unit: '%', width: 90, height: 90, x: 5, y: 5 });
                  }}
                  style={{ maxWidth: '100%', maxHeight: '400px' }}
                />
              </ReactCrop>

              {/* Shape overlay - shows visually on top of crop area */}
              {shapeMask !== 'rect' && imgRef.current && crop.width > 0 && crop.height > 0 && (() => {
                const img = imgRef.current;
                const isPercent = crop.unit === '%';
                const left = isPercent ? (crop.x / 100) * img.width : crop.x;
                const top = isPercent ? (crop.y / 100) * img.height : crop.y;
                const w = isPercent ? (crop.width / 100) * img.width : crop.width;
                const h = isPercent ? (crop.height / 100) * img.height : crop.height;

                return (
                  <svg
                    style={{
                      position: 'absolute',
                      left: left + 'px',
                      top: top + 'px',
                      width: w + 'px',
                      height: h + 'px',
                      pointerEvents: 'none',
                      zIndex: 10,
                    }}
                    viewBox={`0 0 ${w} ${h}`}
                  >
                    <defs>
                      <mask id="shapeMask">
                        <rect width={w} height={h} fill="white" />
                        {shapeMask === 'circle' && (
                          <ellipse cx={w/2} cy={h/2} rx={w/2} ry={h/2} fill="black" />
                        )}
                        {shapeMask === 'triangle' && (
                          <polygon points={`${w/2},0 0,${h} ${w},${h}`} fill="black" />
                        )}
                      </mask>
                    </defs>
                    {/* Dark overlay for areas OUTSIDE the shape */}
                    <rect width={w} height={h} fill="rgba(0,0,0,0.55)" mask="url(#shapeMask)" />
                    {/* Shape border */}
                    {shapeMask === 'circle' && (
                      <ellipse cx={w/2} cy={h/2} rx={w/2 - 1} ry={h/2 - 1} fill="none" stroke="#00e5ff" strokeWidth="2" strokeDasharray="6,4" />
                    )}
                    {shapeMask === 'triangle' && (
                      <polygon points={`${w/2},1 1,${h-1} ${w-1},${h-1}`} fill="none" stroke="#00e5ff" strokeWidth="2" strokeDasharray="6,4" />
                    )}
                  </svg>
                );
              })()}
              </div>

              {/* Compact shape bar + action buttons in one row */}
              <div style={{ display: 'flex', gap: '10px', marginTop: '8px', alignItems: 'center', justifyContent: 'center', flexWrap: 'wrap' }}>
                <div className="shape-bar">
                  <span className="shape-bar-label">Crop:</span>
                  <button className={`shape-btn-sm ${shapeMask === 'rect' ? 'active' : ''}`} onClick={() => setShapeMask('rect')} title="Chữ nhật">
                    <svg width="18" height="18" viewBox="0 0 32 32"><rect x="4" y="6" width="24" height="20" fill="none" stroke="currentColor" strokeWidth="3" rx="2"/></svg>
                  </button>
                  <button className={`shape-btn-sm ${shapeMask === 'circle' ? 'active' : ''}`} onClick={() => setShapeMask('circle')} title="Tròn (biển cấm)">
                    <svg width="18" height="18" viewBox="0 0 32 32"><circle cx="16" cy="16" r="12" fill="none" stroke="currentColor" strokeWidth="3"/></svg>
                  </button>
                  <button className={`shape-btn-sm ${shapeMask === 'triangle' ? 'active' : ''}`} onClick={() => setShapeMask('triangle')} title="Tam giác (biển nguy hiểm)">
                    <svg width="18" height="18" viewBox="0 0 32 32"><polygon points="16,3 2,29 30,29" fill="none" stroke="currentColor" strokeWidth="3" strokeLinejoin="round"/></svg>
                  </button>
                </div>
                <button className="btn-search" onClick={handleSearch} disabled={loading} style={{ fontWeight: 'bold', background: '#e0f7fa' }}>
                  {loading ? <span className="loader">Đang tải...</span> : 'Cắt ảnh & Nhận diện'}
                </button>
                <button className="btn-search" onClick={() => { setPreview(null); setSelectedFile(null); setResults([]); setError(''); }} disabled={loading}>
                  Làm mới
                </button>
              </div>
            </div>
          )}
        </div>

        {error && <div style={{ color: '#ff4d4f', marginTop: '1rem' }}>{error}</div>}

        {results.length > 0 && (
          <section className="results-section">
            <h2 className="results-title">Kết quả tương đồng nhất</h2>
            <div className="cards-container">
              {results.map((item, idx) => (
                <div key={idx} className="result-card">
                  <div className="result-image-wrapper">
                    {item.image_path ? (
                      <img
                        src={`http://localhost:8000/${item.image_path}`}
                        alt={item.label}
                        className="result-image"
                        onError={(e) => {
                          e.target.onerror = null;
                          e.target.src = "https://via.placeholder.com/200?text=Loi+Anh";
                        }}
                      />
                    ) : (
                      <div style={{ color: '#64748b' }}>-</div>
                    )}
                  </div>
                  <div className="result-info">
                    <span className="result-score" style={{ color: 'green', fontWeight: 'bold', marginBottom: '5px', display: 'block' }}>Độ tương đồng: {(item.score * 100).toFixed(1)}%</span>
                    <span className="result-group"><strong>Loại biển:</strong> {formatGroup(item.group)}</span>
                    <span className="result-label"><strong>Ký hiệu (Mã biển):</strong> {item.label}</span>
                    <p className="result-meaning" style={{ marginTop: '5px', marginBottom: '5px' }}>
                      <strong>Chức năng / Mô tả:</strong> {item.meaning && item.meaning !== "nan" && item.meaning !== "Unknown" ? item.meaning : "Đang cập nhật..."}
                    </p>
                    <div style={{ background: '#fffee6', padding: '5px', borderRadius: '4px', marginTop: '5px', borderLeft: '3px solid #ffc107', fontSize: '13px' }}>
                      <strong>Lời khuyên:</strong> {item.advice}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
