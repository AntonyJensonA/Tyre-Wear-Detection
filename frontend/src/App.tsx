import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { UploadCloud, Activity, Car, RefreshCw, DownloadCloud } from 'lucide-react';

interface PredictionResult {
  filename: string;
  wear_detection: {
    status: string;
    mask_area_percentage: number;
  };
  remaining_life: {
    status: string;
    remaining_life_km: number;
  };
}

function App() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [deferredPrompt, setDeferredPrompt] = useState<any>(null);

  useEffect(() => {
    const handleBeforeInstallPrompt = (e: Event) => {
      e.preventDefault();
      setDeferredPrompt(e);
    };
    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
    return () => window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
  }, []);

  const handleInstallClick = async () => {
    if (!deferredPrompt) return;
    deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;
    if (outcome === 'accepted') {
      setDeferredPrompt(null);
    }
  };

  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedImage) return;

    setIsPredicting(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', selectedImage);

    try {
      // Automatically connect to the backend using the current hostname
      const backendUrl = `http://${window.location.hostname}:8000/predict`;
      const response = await axios.post<PredictionResult>(backendUrl, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResult(response.data);
    } catch (err: any) {
      setError(err.message || 'Failed to connect to the prediction server. Make sure the backend is running.');
      console.error(err);
    } finally {
      setIsPredicting(false);
    }
  };

  const resetState = () => {
    setSelectedImage(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  const getWearClass = (percentage: number) => {
    if (percentage < 15) return 'success';
    if (percentage < 25) return 'warning';
    return 'danger';
  };

  const getLifeClass = (km: number) => {
    if (km > 30000) return 'success';
    if (km > 15000) return 'warning';
    return 'danger';
  };

  return (
    <div className="app-container">
      <div>
        <h1>Tyre AI Insight</h1>
        <p className="subtitle">Upload a tyre image to predict wear and remaining lifespan.</p>
        {deferredPrompt && (
          <button 
            onClick={handleInstallClick}
            className="install-button"
            style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', margin: '0 auto 2rem', backgroundColor: '#4ade80', color: '#0f172a' }}
          >
            <DownloadCloud size={20} />
            Install App to Home Screen
          </button>
        )}
      </div>

      <div className="upload-card">
        {!previewUrl ? (
          <label className="file-input-label">
            <UploadCloud className="upload-icon" />
            <span style={{ fontSize: '1.2rem', fontWeight: 600 }}>Click or Drag to Upload Tyre Image</span>
            <span style={{ color: '#94a3b8', fontSize: '0.9rem', marginTop: '0.5rem' }}>JPEG, PNG up to 10MB</span>
            <input 
              type="file" 
              className="file-input" 
              accept="image/*" 
              capture="environment"
              onChange={handleImageChange} 
            />
          </label>
        ) : (
          <div className="preview-container">
            <img src={previewUrl} alt="Tyre preview" className="image-preview" />
            
            <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
              <button 
                onClick={handleUpload} 
                disabled={isPredicting}
                style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
              >
                {isPredicting ? (
                   <RefreshCw className="loading-spinner" size={20} />
                ) : (
                  <Activity size={20} />
                )}
                {isPredicting ? 'Analyzing AI Models...' : 'Run Predictions'}
              </button>
              
              <button 
                onClick={resetState} 
                disabled={isPredicting}
                style={{ backgroundColor: 'transparent', border: '1px solid #475569', color: '#fff' }}
              >
                Clear
              </button>
            </div>
            
            {error && (
              <div style={{ color: '#ef4444', marginTop: '1rem', padding: '1rem', background: 'rgba(239, 68, 68, 0.1)', borderRadius: '8px' }}>
                {error}
              </div>
            )}
          </div>
        )}
      </div>

      {result && (
        <div className="results-grid">
           <div className={`result-card ${getWearClass(result.wear_detection.mask_area_percentage)}`}>
             <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="result-label">Tear Wear Area</span>
                <Activity size={20} style={{ color: '#94a3b8' }} />
             </div>
             <div className="result-value">
               {result.wear_detection.mask_area_percentage}%
             </div>
             <p style={{ margin: 0, fontSize: '0.85rem', color: '#94a3b8' }}>
               Percentage of tyre surface exhibiting wear
             </p>
           </div>
           
           <div className={`result-card ${getLifeClass(result.remaining_life.remaining_life_km)}`}>
             <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="result-label">Remaining Life</span>
                <Car size={20} style={{ color: '#94a3b8' }} />
             </div>
             <div className="result-value">
               {result.remaining_life.remaining_life_km.toLocaleString()} km
             </div>
             <p style={{ margin: 0, fontSize: '0.85rem', color: '#94a3b8' }}>
               Estimated safe driving distance remaining
             </p>
           </div>
        </div>
      )}
    </div>
  );
}

export default App;
