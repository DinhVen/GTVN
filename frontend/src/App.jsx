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
      // If user didn't crop properly, just return original file
      return selectedFile;
    }

    const canvas = document.createElement('canvas');
    const scaleX = image.naturalWidth / image.width;
    const scaleY = image.naturalHeight / image.height;

    canvas.width = completedCrop.width * scaleX;
    canvas.height = completedCrop.height * scaleY;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(
      image,
      completedCrop.x * scaleX,
      completedCrop.y * scaleY,
      completedCrop.width * scaleX,
      completedCrop.height * scaleY,
      0,
      0,
      canvas.width,
      canvas.height
    );

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

  const getActionAdvice = (group) => {
    switch (group?.toLowerCase()) {
      case 'cấm':
        return "Tuyệt đối không được thực hiện hành vi bị cấm, vi phạm sẽ bị xử phạt hành chính.";
      case 'nguy hiểm':
        return "Giảm tốc độ, chú ý quan sát xung quanh và chuẩn bị ứng phó với tình huống xấu.";
      case 'hiệu lệnh':
        return "Bắt buộc phải tuân theo hướng dẫn hoặc chỉ thị của biển báo này.";
      case 'chỉ dẫn':
        return "Theo dõi để lấy thông tin định hướng đường đi, giúp di chuyển thuận lợi.";
      default:
        return "Hãy tuân thủ luật giao thông chung.";
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
                * Hãy kéo chọn VÙNG CHỨA BIỂN BÁO để loại bỏ nền/chữ, kết quả sẽ chính xác hơn!
              </p>
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
                  onLoad={(e) => {
                    const { width, height } = e.currentTarget;
                    setCrop({ unit: '%', width: 90, height: 90, x: 5, y: 5 });
                  }}
                  style={{ maxWidth: '100%', maxHeight: '400px' }}
                />
              </ReactCrop>

              <div style={{ display: 'flex', gap: '1rem', marginTop: '10px' }}>
                <button
                  className="btn-search"
                  onClick={handleSearch}
                  disabled={loading}
                  style={{ fontWeight: 'bold', background: '#e0f7fa' }}
                >
                  {loading ? <span className="loader">Đang tải...</span> : 'Cắt ảnh & Nhận diện'}
                </button>
                <button
                  className="btn-search"
                  onClick={() => {
                    setPreview(null);
                    setSelectedFile(null);
                    setResults([]);
                    setError('');
                  }}
                  disabled={loading}
                >
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
