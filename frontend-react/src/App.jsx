import { useState, useEffect, useRef, useCallback } from 'react';
import './index.css';

// API Configuration
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

function App() {
    // State
    const [isStreaming, setIsStreaming] = useState(false);
    const [isConnected, setIsConnected] = useState(false);
    const [metrics, setMetrics] = useState({
        fps: 0,
        latency: 0,
        detections: 0,
        driftScore: 0,
    });
    const [driftStatus, setDriftStatus] = useState('normal');
    const [confidence, setConfidence] = useState(0.5);
    const [modelInfo, setModelInfo] = useState(null);
    const [error, setError] = useState(null);

    // Refs
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const overlayCanvasRef = useRef(null);
    const wsRef = useRef(null);
    const streamRef = useRef(null);
    const frameIntervalRef = useRef(null);
    const fpsHistoryRef = useRef([]);

    // Fetch model info on mount
    useEffect(() => {
        fetchModelInfo();
    }, []);

    const fetchModelInfo = async () => {
        try {
            const response = await fetch(`${API_URL}/model-info`);
            if (response.ok) {
                const data = await response.json();
                setModelInfo(data);
                setIsConnected(true);
            }
        } catch (err) {
            console.error('Failed to connect to backend:', err);
            setError('Backend not connected. Make sure the API is running.');
        }
    };

    // Start webcam stream
    const startWebcam = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: 'user' },
                audio: false,
            });
            streamRef.current = stream;
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
            }
            return true;
        } catch (err) {
            console.error('Webcam access denied:', err);
            setError('Could not access webcam. Please allow camera permissions.');
            return false;
        }
    };

    // Stop webcam stream
    const stopWebcam = () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }
    };

    // Connect WebSocket
    const connectWebSocket = () => {
        wsRef.current = new WebSocket(`${WS_URL}/ws/detect`);

        wsRef.current.onopen = () => {
            console.log('WebSocket connected');
            setIsConnected(true);
        };

        wsRef.current.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleDetectionResult(data);
        };

        wsRef.current.onerror = (err) => {
            console.error('WebSocket error:', err);
            setError('WebSocket connection failed');
        };

        wsRef.current.onclose = () => {
            console.log('WebSocket closed');
            setIsConnected(false);
        };
    };

    // Handle detection results
    const handleDetectionResult = (data) => {
        const now = performance.now();
        fpsHistoryRef.current.push(now);

        // Keep only last 30 frames for FPS calculation
        const oneSecondAgo = now - 1000;
        fpsHistoryRef.current = fpsHistoryRef.current.filter(t => t > oneSecondAgo);

        const fps = fpsHistoryRef.current.length;

        setMetrics({
            fps: fps,
            latency: data.inference_time_ms || 0,
            detections: data.count || 0,
            driftScore: data.drift_score || 0,
        });

        // Update drift status
        if (data.drift_score > 3) {
            setDriftStatus('alert');
        } else if (data.drift_score > 2) {
            setDriftStatus('warning');
        } else {
            setDriftStatus('normal');
        }

        // Draw bounding boxes
        drawDetections(data.detections || []);
    };

    // Draw bounding boxes on canvas
    const drawDetections = (detections) => {
        const canvas = overlayCanvasRef.current;
        const video = videoRef.current;
        if (!canvas || !video) return;

        const ctx = canvas.getContext('2d');
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        detections.forEach(det => {
            const [x1, y1, x2, y2] = det.bbox;
            const conf = det.confidence;

            // Draw box
            ctx.strokeStyle = '#38ef7d';
            ctx.lineWidth = 3;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            // Draw label background
            const label = `person ${(conf * 100).toFixed(0)}%`;
            ctx.font = 'bold 14px Inter';
            const textWidth = ctx.measureText(label).width;

            ctx.fillStyle = 'rgba(56, 239, 125, 0.9)';
            ctx.fillRect(x1, y1 - 25, textWidth + 10, 22);

            // Draw label text
            ctx.fillStyle = '#0a0a1a';
            ctx.fillText(label, x1 + 5, y1 - 8);
        });
    };

    // Capture and send frame
    const captureAndSendFrame = useCallback(() => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas || !wsRef.current) return;
        if (wsRef.current.readyState !== WebSocket.OPEN) return;

        const ctx = canvas.getContext('2d');
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        ctx.drawImage(video, 0, 0);

        // Convert to base64 and send
        const base64 = canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
        wsRef.current.send(JSON.stringify({
            image: base64,
            confidence: confidence,
        }));
    }, [confidence]);

    // Start detection
    const startDetection = async () => {
        setError(null);
        const webcamStarted = await startWebcam();
        if (!webcamStarted) return;

        connectWebSocket();
        setIsStreaming(true);

        // Wait for video to be ready
        setTimeout(() => {
            frameIntervalRef.current = setInterval(captureAndSendFrame, 100); // 10 FPS
        }, 500);
    };

    // Stop detection
    const stopDetection = () => {
        setIsStreaming(false);

        if (frameIntervalRef.current) {
            clearInterval(frameIntervalRef.current);
            frameIntervalRef.current = null;
        }

        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }

        stopWebcam();
        fpsHistoryRef.current = [];

        // Clear overlay
        const canvas = overlayCanvasRef.current;
        if (canvas) {
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    };

    // Get status text
    const getStatusText = () => {
        switch (driftStatus) {
            case 'alert':
                return '⚠️ DRIFT DETECTED - Input distribution shifted significantly!';
            case 'warning':
                return '⚡ Drift Warning - Minor distribution shift detected';
            default:
                return '✅ System Normal - No drift detected';
        }
    };

    return (
        <div className="app-container">
            {/* Header */}
            <header className="header">
                <h1>🎯 Real-Time Detection</h1>
                <p className="subtitle">
                    YOLOv8 with MLOps Monitoring
                    <span className="live-badge">LIVE</span>
                </p>
            </header>

            {/* Error Banner */}
            {error && (
                <div className="glass-card" style={{ marginBottom: '20px', borderColor: 'rgba(255, 107, 107, 0.5)' }}>
                    <p style={{ color: '#ff6b6b' }}>⚠️ {error}</p>
                </div>
            )}

            {/* Main Content */}
            <div className="main-content">
                {/* Video Section */}
                <div className="glass-card">
                    <h2 className="card-title">
                        <span className="icon">📹</span>
                        Live Detection
                    </h2>

                    <div className="video-container">
                        {!isStreaming ? (
                            <div className="placeholder">
                                <span className="icon">🎥</span>
                                <span className="text">Click "Start Detection" to begin</span>
                            </div>
                        ) : (
                            <>
                                <video ref={videoRef} autoPlay playsInline muted />
                                <canvas ref={overlayCanvasRef} style={{
                                    position: 'absolute',
                                    top: 0,
                                    left: 0,
                                    width: '100%',
                                    height: '100%',
                                    pointerEvents: 'none'
                                }} />
                                <div className="video-overlay">
                                    <div className="video-stat">
                                        FPS: <span className="value">{metrics.fps}</span>
                                    </div>
                                    <div className="video-stat">
                                        Latency: <span className="value">{metrics.latency.toFixed(0)}ms</span>
                                    </div>
                                    <div className="video-stat">
                                        Objects: <span className="value">{metrics.detections}</span>
                                    </div>
                                </div>
                            </>
                        )}
                        <canvas ref={canvasRef} style={{ display: 'none' }} />
                    </div>

                    {/* Controls */}
                    <div className="controls">
                        {!isStreaming ? (
                            <button className="btn btn-primary" onClick={startDetection}>
                                🚀 Start Detection
                            </button>
                        ) : (
                            <button className="btn btn-danger" onClick={stopDetection}>
                                ⏹️ Stop Detection
                            </button>
                        )}
                    </div>

                    {/* Confidence Slider */}
                    <div className="slider-container">
                        <div className="slider-label">
                            <span>Confidence Threshold</span>
                            <span>{(confidence * 100).toFixed(0)}%</span>
                        </div>
                        <input
                            type="range"
                            className="slider"
                            min="0.1"
                            max="0.9"
                            step="0.05"
                            value={confidence}
                            onChange={(e) => setConfidence(parseFloat(e.target.value))}
                        />
                    </div>
                </div>

                {/* Sidebar */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                    {/* Connection Status */}
                    <div className="glass-card">
                        <div className="connection-status">
                            <span className={`connection-dot ${isConnected ? 'connected' : 'disconnected'}`}></span>
                            <span>{isConnected ? 'Backend Connected' : 'Backend Disconnected'}</span>
                        </div>
                    </div>

                    {/* Metrics */}
                    <div className="glass-card">
                        <h3 className="card-title">
                            <span className="icon">📊</span>
                            Live Metrics
                        </h3>
                        <div className="metrics-grid">
                            <div className="metric-card">
                                <div className="emoji">⚡</div>
                                <div className="value">{metrics.fps}</div>
                                <div className="label">FPS</div>
                            </div>
                            <div className="metric-card">
                                <div className="emoji">⏱️</div>
                                <div className="value">{metrics.latency.toFixed(0)}</div>
                                <div className="label">Latency (ms)</div>
                            </div>
                            <div className="metric-card">
                                <div className="emoji">👥</div>
                                <div className="value">{metrics.detections}</div>
                                <div className="label">Detections</div>
                            </div>
                            <div className="metric-card">
                                <div className="emoji">📉</div>
                                <div className="value">{metrics.driftScore.toFixed(2)}</div>
                                <div className="label">Drift Score</div>
                            </div>
                        </div>

                        {/* Status */}
                        <div className={`status-indicator status-${driftStatus}`}>
                            {getStatusText()}
                        </div>
                    </div>

                    {/* Model Info */}
                    <div className="glass-card">
                        <h3 className="card-title">
                            <span className="icon">🤖</span>
                            Model Info
                        </h3>
                        {modelInfo ? (
                            <div className="model-info">
                                <div className="info-row">
                                    <span className="key">Model</span>
                                    <span className="val">{modelInfo.model_name || 'YOLOv8n'}</span>
                                </div>
                                <div className="info-row">
                                    <span className="key">Dataset</span>
                                    <span className="val">CrowdHuman</span>
                                </div>
                                <div className="info-row">
                                    <span className="key">mAP@50</span>
                                    <span className="val">66.2%</span>
                                </div>
                                <div className="info-row">
                                    <span className="key">Classes</span>
                                    <span className="val">person</span>
                                </div>
                            </div>
                        ) : (
                            <p style={{ color: 'var(--text-muted)' }}>Loading model info...</p>
                        )}
                    </div>
                </div>
            </div>

            {/* Footer */}
            <footer className="footer">
                <p>
                    Built with YOLOv8, FastAPI, React & WebRTC |{' '}
                    <a href="https://github.com/raghavendra-24/realtime-detection-mlops" target="_blank" rel="noopener noreferrer">
                        GitHub
                    </a>
                </p>
            </footer>
        </div>
    );
}

export default App;
