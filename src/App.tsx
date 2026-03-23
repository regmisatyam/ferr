import { useEffect, useRef, useState } from 'react';
import { initFaceLandmarker, detectLandmarks } from './vision/faceLandmarker';
import { extractFeatures, featuresToArray, normalizeFeatures } from './features/features';
import { renderHeatmap } from './heatmap/heatmap';
import { renderExpressionHeatmap, renderExpressionBars } from './heatmap/expressionHeatmap';
import { OnlineSoftmaxClassifier } from './learning/onlineSoftmax';
import { 
  initDB, 
  saveBaseline, 
  loadBaseline, 
  saveSample,
  clearSamples,
  clearClassifierState,
  exportData,
  importData,
  saveRegisteredFaces,
  loadRegisteredFaces,
  clearRegisteredFaces
} from './storage/indexedDb';
import {
  initFaceRecognition,
  extractFaceDescriptor,
  recognizeFace,
  registerFace,
  autoImproveRecognition,
  createHybridFeatures,
  detectExpressions
} from './recognition/faceRecognition';
import { Landmark, EmotionLabel, Prediction, RegisteredFace, FaceRecognitionResult, ExpressionResult } from './types';
import './App.css';

const EMOTIONS: EmotionLabel[] = ['Neutral', 'Happy', 'Angry', 'Sad', 'Surprise', 'Fear', 'Disgust'];
const BASELINE_FRAMES = 30;

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();
  const classifierRef = useRef<OnlineSoftmaxClassifier | null>(null);

  const [isInitialized, setIsInitialized] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [error, setError] = useState<string>('');
  const [baseline, setBaseline] = useState<Landmark[] | null>(null);
  const [isCapturingBaseline, setIsCapturingBaseline] = useState(false);
  const [baselineProgress, setBaselineProgress] = useState(0);
  const [currentLandmarks, setCurrentLandmarks] = useState<Landmark[] | null>(null);
  const [tensionScore, setTensionScore] = useState(0);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [sampleCounts, setSampleCounts] = useState<Record<EmotionLabel, number>>({
    Neutral: 0,
    Happy: 0,
    Angry: 0,
    Sad: 0,
    Surprise: 0,
    Fear: 0,
    Disgust: 0
  });
  const [showDebug, setShowDebug] = useState(false);
  const [fps, setFps] = useState(0);
  
  // Face recognition state
  const [recognitionReady, setRecognitionReady] = useState(false);
  const [registeredFaces, setRegisteredFaces] = useState<RegisteredFace[]>([]);
  const [faceRecognition, setFaceRecognition] = useState<FaceRecognitionResult | null>(null);
  const [newPersonName, setNewPersonName] = useState('');
  const [useHybridModel, setUseHybridModel] = useState(false);
  const lastRecognitionUpdate = useRef<number>(0);
  
  // Expression detection state
  const [pretrainedExpression, setPretrainedExpression] = useState<ExpressionResult | null>(null);
  const [autoTrain, setAutoTrain] = useState(false);
  const [showExpressionHeatmap, setShowExpressionHeatmap] = useState(true);
  const [showExpressionBars, setShowExpressionBars] = useState(true);
  const lastAutoTrainUpdate = useRef<number>(0);
  const autoTrainCount = useRef<number>(0);
  
  // Interactive training state
  const [interactiveTraining, setInteractiveTraining] = useState(true);
  const [showEmotionPrompt, setShowEmotionPrompt] = useState(false);
  const [promptedEmotion, setPromptedEmotion] = useState<EmotionLabel | null>(null);
  const lastEmotionRef = useRef<EmotionLabel | null>(null);
  const emotionStableCount = useRef<number>(0);
  const lastPromptTime = useRef<number>(0);
  const interactiveTrainCount = useRef<number>(0);

  // Initialize
  useEffect(() => {
    async function init() {
      try {
        // Initialize IndexedDB
        await initDB();
        
        // Initialize FaceLandmarker
        await initFaceLandmarker();
        
        // Initialize face recognition
        await initFaceRecognition();
        setRecognitionReady(true);
        
        // Load registered faces
        const savedFaces = await loadRegisteredFaces();
        setRegisteredFaces(savedFaces);
        
        // Initialize classifier (hybrid if we have face recognition)
        const numFeatures = 11; // Base geometric features
        classifierRef.current = new OnlineSoftmaxClassifier(numFeatures, false);
        
        // Load saved baseline
        const savedBaseline = await loadBaseline();
        if (savedBaseline) {
          setBaseline(savedBaseline);
        }
        
        // Load classifier state
        const loaded = await classifierRef.current.load();
        if (loaded) {
          setSampleCounts(classifierRef.current.getSampleCounts());
        }
        
        setIsInitialized(true);
      } catch (err) {
        setError(`Initialization failed: ${err}`);
        console.error(err);
      }
    }

    init();
  }, []);

  // Start camera
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        }
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play();
          setCameraActive(true);
          startProcessing();
        };
      }
    } catch (err) {
      setError(`Camera access failed: ${err}`);
      console.error(err);
    }
  };

  // Stop camera
  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
  };

  // Process video frames
  const startProcessing = () => {
    let frameCount = 0;
    let fpsLastUpdate = performance.now();

    const processFrame = () => {
      if (!videoRef.current || !canvasRef.current || !isInitialized) {
        animationFrameRef.current = requestAnimationFrame(processFrame);
        return;
      }

      const now = performance.now();
      const timestamp = now;

      // Detect landmarks
      const landmarks = detectLandmarks(videoRef.current, timestamp);
      
      if (landmarks) {
        setCurrentLandmarks(landmarks);

        // Render heatmap
        const tension = renderHeatmap(
          canvasRef.current,
          videoRef.current,
          landmarks,
          baseline,
          showDebug
        );
        setTensionScore(tension);
        
        // Detect expressions using pre-trained model (throttled to every 200ms)
        if (recognitionReady && videoRef.current && now - lastRecognitionUpdate.current > 200) {
          lastRecognitionUpdate.current = now;
          
          detectExpressions(videoRef.current).then(expressionResult => {
            if (expressionResult) {
              setPretrainedExpression(expressionResult);
              
              // Render expression heatmap
              if (showExpressionHeatmap && canvasRef.current && videoRef.current) {
                renderExpressionHeatmap(
                  canvasRef.current,
                  videoRef.current,
                  expressionResult.expressions
                );
              }
              
              // Render expression bars
              if (showExpressionBars && canvasRef.current) {
                renderExpressionBars(
                  canvasRef.current,
                  expressionResult.expressions
                );
              }
              
              // Interactive training: detect mood changes
              if (interactiveTraining && !showEmotionPrompt) {
                const currentEmotion = expressionResult.dominantEmotion;
                
                // Check if emotion has changed
                if (lastEmotionRef.current !== null && 
                    lastEmotionRef.current !== currentEmotion) {
                  emotionStableCount.current = 0;
                } else if (lastEmotionRef.current === currentEmotion) {
                  emotionStableCount.current++;
                }
                
                lastEmotionRef.current = currentEmotion;
                
                // If emotion has been stable for 5 frames (1 second) and hasn't prompted recently
                if (emotionStableCount.current === 5 && 
                    now - lastPromptTime.current > 10000 && // 10 seconds between prompts
                    expressionResult.confidence > 0.6) {
                  
                  // Check if there's a discrepancy with trained model
                  const shouldPrompt = !prediction || 
                    (classifierRef.current && classifierRef.current.getTotalSamples() > 10 &&
                     prediction.label !== currentEmotion);
                  
                  if (shouldPrompt || Math.random() < 0.3) { // Also randomly prompt 30% of the time
                    setPromptedEmotion(currentEmotion);
                    setShowEmotionPrompt(true);
                    lastPromptTime.current = now;
                    emotionStableCount.current = 0;
                  }
                }
              }
              
              // Auto-train if enabled and confident (throttled to every 2 seconds)
              if (autoTrain && 
                  expressionResult.confidence > 0.7 && 
                  currentLandmarks &&
                  videoRef.current &&
                  now - lastAutoTrainUpdate.current > 2000) {
                lastAutoTrainUpdate.current = now;
                
                // Auto-label with pre-trained model's prediction
                const features = extractFeatures(currentLandmarks);
                let featureArray = normalizeFeatures(featuresToArray(features));
                
                if (useHybridModel) {
                  extractFaceDescriptor(videoRef.current).then(descriptor => {
                    featureArray = createHybridFeatures(featureArray, descriptor);
                    if (classifierRef.current) {
                      classifierRef.current.resizeForHybrid(featureArray.length);
                      classifierRef.current.train(featureArray, expressionResult.dominantEmotion).then(() => {
                        setSampleCounts(classifierRef.current!.getSampleCounts());
                        autoTrainCount.current++;
                      });
                    }
                  });
                } else {
                  if (classifierRef.current) {
                    classifierRef.current.train(featureArray, expressionResult.dominantEmotion).then(() => {
                      setSampleCounts(classifierRef.current!.getSampleCounts());
                      autoTrainCount.current++;
                    });
                  }
                }
                
                // Save sample
                saveSample({
                  timestamp: Date.now(),
                  features: featureArray,
                  label: expressionResult.dominantEmotion
                });
              }
            } else {
              setPretrainedExpression(null);
            }
          });
        }

        // Face recognition (separate from expression detection)
        if (recognitionReady) {
          extractFaceDescriptor(videoRef.current).then(descriptor => {
            if (descriptor) {
              const recognition = recognizeFace(descriptor, registeredFaces);
              setFaceRecognition(recognition);
              
              // Auto-improve recognition if confident
              if (recognition && recognition.confidence > 0.85) {
                const improved = autoImproveRecognition(descriptor, recognition, registeredFaces, 0.85);
                if (improved !== registeredFaces) {
                  setRegisteredFaces(improved);
                  saveRegisteredFaces(improved);
                }
              }
            } else {
              setFaceRecognition(null);
            }
          });
        }
        
        // Extract features and predict with YOUR trained model (runs every frame)
        if (classifierRef.current && classifierRef.current.getTotalSamples() > 0) {
          const features = extractFeatures(landmarks);
          let featureArray = normalizeFeatures(featuresToArray(features));
          
          // Use hybrid features if enabled
          if (useHybridModel && recognitionReady) {
            extractFaceDescriptor(videoRef.current).then(descriptor => {
              const hybridFeatures = createHybridFeatures(featureArray, descriptor);
              if (classifierRef.current && classifierRef.current.getTotalSamples() > 0) {
                classifierRef.current.resizeForHybrid(hybridFeatures.length);
                const pred = classifierRef.current.predict(hybridFeatures);
                setPrediction(pred);
              }
            });
          } else {
            // Predict with geometric features only
            const pred = classifierRef.current.predict(featureArray);
            setPrediction(pred);
          }
        } else {
          // No trained samples yet, clear prediction
          setPrediction(null);
        }
      } else {
        setCurrentLandmarks(null);
        
        // Still draw video frame
        if (canvasRef.current && videoRef.current) {
          const ctx = canvasRef.current.getContext('2d');
          if (ctx) {
            const width = videoRef.current.videoWidth;
            const height = videoRef.current.videoHeight;
            if (canvasRef.current.width !== width || canvasRef.current.height !== height) {
              canvasRef.current.width = width;
              canvasRef.current.height = height;
            }
            ctx.drawImage(videoRef.current, 0, 0, width, height);
          }
        }
      }

      // Calculate FPS
      frameCount++;
      if (now - fpsLastUpdate > 1000) {
        setFps(Math.round(frameCount * 1000 / (now - fpsLastUpdate)));
        frameCount = 0;
        fpsLastUpdate = now;
      }

      animationFrameRef.current = requestAnimationFrame(processFrame);
    };

    processFrame();
  };

  // Capture baseline
  const captureBaseline = async () => {
    if (!currentLandmarks) {
      setError('No face detected. Please ensure your face is visible.');
      return;
    }

    setIsCapturingBaseline(true);
    setBaselineProgress(0);
    
    const collectedFrames: Landmark[][] = [];

    const collectFrame = () => {
      if (currentLandmarks) {
        collectedFrames.push([...currentLandmarks]);
        setBaselineProgress(collectedFrames.length);

        if (collectedFrames.length < BASELINE_FRAMES) {
          setTimeout(collectFrame, 100); // Collect every 100ms
        } else {
          // Average all frames
          const avgLandmarks: Landmark[] = [];
          const numLandmarks = collectedFrames[0].length;

          for (let i = 0; i < numLandmarks; i++) {
            let sumX = 0, sumY = 0, sumZ = 0;
            for (const frame of collectedFrames) {
              sumX += frame[i].x;
              sumY += frame[i].y;
              sumZ += frame[i].z;
            }
            avgLandmarks.push({
              x: sumX / BASELINE_FRAMES,
              y: sumY / BASELINE_FRAMES,
              z: sumZ / BASELINE_FRAMES
            });
          }

          // Save baseline
          saveBaseline(avgLandmarks).then(() => {
            setBaseline(avgLandmarks);
            setIsCapturingBaseline(false);
            setBaselineProgress(0);
          });
        }
      } else {
        setIsCapturingBaseline(false);
        setError('Lost face detection during baseline capture');
      }
    };

    collectFrame();
  };

  // Label current expression
  const labelExpression = async (label: EmotionLabel, source: string = 'manual') => {
    if (!currentLandmarks) {
      setError('No face detected');
      return;
    }

    if (!classifierRef.current) {
      setError('Classifier not initialized');
      return;
    }

    try {
      const features = extractFeatures(currentLandmarks);
      let featureArray = normalizeFeatures(featuresToArray(features));
      
      // Use hybrid features if enabled and available
      if (useHybridModel && recognitionReady && videoRef.current) {
        const descriptor = await extractFaceDescriptor(videoRef.current);
        featureArray = createHybridFeatures(featureArray, descriptor);
        classifierRef.current.resizeForHybrid(featureArray.length);
      }

      // Save sample
      await saveSample({
        timestamp: Date.now(),
        features: featureArray,
        label
      });

      // Train classifier
      await classifierRef.current.train(featureArray, label);
      setSampleCounts(classifierRef.current.getSampleCounts());

      if (source === 'interactive') {
        interactiveTrainCount.current++;
      }

      console.log(`Labeled as ${label} (${source}), total samples: ${classifierRef.current.getTotalSamples()}`);
    } catch (err) {
      setError(`Failed to label: ${err}`);
      console.error(err);
    }
  };
  
  // Handle interactive prompt response
  const handlePromptResponse = async (userEmotion: EmotionLabel | 'none') => {
    setShowEmotionPrompt(false);
    
    if (userEmotion !== 'none' && currentLandmarks) {
      // User confirmed an emotion - train with their response
      await labelExpression(userEmotion, 'interactive');
    }
    
    setPromptedEmotion(null);
  };

  // Register face
  const handleRegisterFace = async () => {
    if (!videoRef.current || !recognitionReady) {
      setError('Face recognition not ready');
      return;
    }

    if (!newPersonName.trim()) {
      setError('Please enter a name');
      return;
    }

    try {
      const updated = await registerFace(videoRef.current, newPersonName.trim(), registeredFaces);
      if (updated) {
        setRegisteredFaces(updated);
        await saveRegisteredFaces(updated);
        setNewPersonName('');
        console.log(`Registered face for: ${newPersonName}`);
      } else {
        setError('No face detected for registration');
      }
    } catch (err) {
      setError(`Failed to register face: ${err}`);
    }
  };

  // Reset learning
  const resetLearning = async () => {
    if (!classifierRef.current) return;

    if (confirm('Are you sure you want to reset all learning data?')) {
      try {
        await clearSamples();
        await clearClassifierState();
        classifierRef.current.reset();
        setSampleCounts(classifierRef.current.getSampleCounts());
        setPrediction(null);
      } catch (err) {
        setError(`Failed to reset: ${err}`);
      }
    }
  };
  
  // Reset face recognition
  const resetFaceRecognition = async () => {
    if (confirm('Are you sure you want to delete all registered faces?')) {
      try {
        await clearRegisteredFaces();
        setRegisteredFaces([]);
        setFaceRecognition(null);
      } catch (err) {
        setError(`Failed to reset face recognition: ${err}`);
      }
    }
  };

  // Export data
  const handleExport = async () => {
    try {
      const data = await exportData();
      const blob = new Blob([data], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `face-expression-data-${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(`Export failed: ${err}`);
    }
  };

  // Import data
  const handleImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      await importData(text);
      
      // Reload state
      const savedBaseline = await loadBaseline();
      if (savedBaseline) {
        setBaseline(savedBaseline);
      }
      
      if (classifierRef.current) {
        await classifierRef.current.load();
        setSampleCounts(classifierRef.current.getSampleCounts());
      }
      
      const savedFaces = await loadRegisteredFaces();
      setRegisteredFaces(savedFaces);

      alert('Data imported successfully!');
    } catch (err) {
      setError(`Import failed: ${err}`);
    }
    
    // Reset file input
    event.target.value = '';
  };

  return (
    <div className="app">
      <header>
        <h1>Face Expression + Tension Heatmap + Continual Learning</h1>
        {pretrainedExpression && (
          <div className="header-emotion">
            Current: <span className="header-emotion-label">{pretrainedExpression.dominantEmotion}</span>
            <span className="header-emoji">
              {pretrainedExpression.dominantEmotion === 'Happy' && '😊'}
              {pretrainedExpression.dominantEmotion === 'Sad' && '😢'}
              {pretrainedExpression.dominantEmotion === 'Angry' && '😠'}
              {pretrainedExpression.dominantEmotion === 'Surprise' && '😲'}
              {pretrainedExpression.dominantEmotion === 'Fear' && '😨'}
              {pretrainedExpression.dominantEmotion === 'Disgust' && '🤢'}
              {pretrainedExpression.dominantEmotion === 'Neutral' && '😐'}
            </span>
            <span className="header-confidence">
              {(pretrainedExpression.confidence * 100).toFixed(0)}%
            </span>
          </div>
        )}
      </header>

      <div className="container">
        <div className="video-section">
          <div className="video-container">
            <video ref={videoRef} style={{ display: 'none' }} />
            <canvas ref={canvasRef} />
            {!currentLandmarks && cameraActive && (
              <div className="warning">⚠️ No face detected</div>
            )}
          </div>

          <div className="controls">
            {!cameraActive ? (
              <button onClick={startCamera} disabled={!isInitialized}>
                Start Camera
              </button>
            ) : (
              <button onClick={stopCamera}>Stop Camera</button>
            )}
            
            <button 
              onClick={captureBaseline} 
              disabled={!cameraActive || isCapturingBaseline || !currentLandmarks}
            >
              {isCapturingBaseline 
                ? `Capturing... ${baselineProgress}/${BASELINE_FRAMES}`
                : 'Capture Neutral Baseline'}
            </button>

            <label>
              <input 
                type="checkbox" 
                checked={showDebug} 
                onChange={(e) => setShowDebug(e.target.checked)}
              />
              Debug Mode
            </label>
            
            <label>
              <input 
                type="checkbox" 
                checked={useHybridModel} 
                onChange={(e) => setUseHybridModel(e.target.checked)}
              />
              Hybrid Model
            </label>
            
            <label>
              <input 
                type="checkbox" 
                checked={interactiveTraining} 
                onChange={(e) => setInteractiveTraining(e.target.checked)}
              />
              Interactive Training ({interactiveTrainCount.current})
            </label>
            
            <label>
              <input 
                type="checkbox" 
                checked={autoTrain} 
                onChange={(e) => setAutoTrain(e.target.checked)}
              />
              Auto-Train ({autoTrainCount.current})
            </label>
            
            <label>
              <input 
                type="checkbox" 
                checked={showExpressionHeatmap} 
                onChange={(e) => setShowExpressionHeatmap(e.target.checked)}
              />
              Expression Heatmap
            </label>
            
            <label>
              <input 
                type="checkbox" 
                checked={showExpressionBars} 
                onChange={(e) => setShowExpressionBars(e.target.checked)}
              />
              Expression Bars
            </label>
          </div>
        </div>

        <div className="stats-section">
          <div className="stats-panel">
            <h2>Status</h2>
            <div className="stat">
              <span>Initialized:</span>
              <span>{isInitialized ? '✓' : '...'}</span>
            </div>
            <div className="stat">
              <span>Camera:</span>
              <span>{cameraActive ? '✓' : '✗'}</span>
            </div>
            <div className="stat">
              <span>Face Detected:</span>
              <span>{currentLandmarks ? '✓' : '✗'}</span>
            </div>
            <div className="stat">
              <span>Baseline:</span>
              <span>{baseline ? '✓ Captured' : '✗ Missing'}</span>
            </div>
            <div className="stat">
              <span>FPS:</span>
              <span>{fps}</span>
            </div>
            <div className="stat">
              <span>Trained Samples:</span>
              <span>{classifierRef.current?.getTotalSamples() || 0}</span>
            </div>
            {prediction && (
              <div className="stat">
                <span>Model Active:</span>
                <span>✓ Predicting</span>
              </div>
            )}
          </div>

          <div className="stats-panel">
            <h2>Pre-trained Model</h2>
            {pretrainedExpression ? (
              <div className="current-expression">
                <div className="expression-emoji">
                  {pretrainedExpression.dominantEmotion === 'Happy' && '😊'}
                  {pretrainedExpression.dominantEmotion === 'Sad' && '😢'}
                  {pretrainedExpression.dominantEmotion === 'Angry' && '😠'}
                  {pretrainedExpression.dominantEmotion === 'Surprise' && '😲'}
                  {pretrainedExpression.dominantEmotion === 'Fear' && '😨'}
                  {pretrainedExpression.dominantEmotion === 'Disgust' && '🤢'}
                  {pretrainedExpression.dominantEmotion === 'Neutral' && '😐'}
                </div>
                <div className="expression-label">{pretrainedExpression.dominantEmotion}</div>
                <div className="expression-confidence">
                  {(pretrainedExpression.confidence * 100).toFixed(1)}% confident
                </div>
              </div>
            ) : (
              <div className="info">Detecting expressions...</div>
            )}
          </div>

          <div className="stats-panel">
            <h2>Your Trained Model</h2>
            {classifierRef.current && classifierRef.current.getTotalSamples() > 0 ? (
              <>
                {prediction ? (
                  <>
                    <div className="current-expression">
                      <div className="expression-emoji">
                        {prediction.label === 'Happy' && '😊'}
                        {prediction.label === 'Sad' && '😢'}
                        {prediction.label === 'Angry' && '😠'}
                        {prediction.label === 'Surprise' && '😲'}
                        {prediction.label === 'Fear' && '😨'}
                        {prediction.label === 'Disgust' && '🤢'}
                        {prediction.label === 'Neutral' && '😐'}
                      </div>
                      <div className="expression-label">{prediction.label}</div>
                      <div className="expression-confidence">
                        {(prediction.confidence * 100).toFixed(1)}% confident
                      </div>
                    </div>
                    {pretrainedExpression && prediction.label !== pretrainedExpression.dominantEmotion && (
                      <div className="model-discrepancy">
                        ⚠️ Differs from pre-trained model
                      </div>
                    )}
                  </>
                ) : (
                  <div className="info">Processing... {classifierRef.current.getTotalSamples()} samples loaded</div>
                )}
              </>
            ) : (
              <div className="info">
                {interactiveTraining ? 'Answer prompts to train...' : 
                 autoTrain ? 'Auto-training in progress...' : 
                 'Label expressions or enable training'}
              </div>
            )}
          </div>

          <div className="stats-panel">
            <h2>Tension Score</h2>
            <div className="tension-score">
              {tensionScore.toFixed(1)}
            </div>
            {!baseline && (
              <div className="info">Capture baseline first</div>
            )}
          </div>

          <div className="stats-panel">
            <h2>Face Recognition</h2>
            {recognitionReady ? (
              <>
                {faceRecognition ? (
                  <div className="recognition-result">
                    <div className="recognition-name">{faceRecognition.name}</div>
                    <div className="recognition-confidence">
                      {(faceRecognition.confidence * 100).toFixed(1)}% confident
                    </div>
                    <div className="recognition-distance">
                      Distance: {faceRecognition.distance.toFixed(3)}
                    </div>
                  </div>
                ) : (
                  <div className="info">
                    {registeredFaces.length > 0 ? 'Unknown person' : 'No faces registered'}
                  </div>
                )}
                
                <div className="face-register">
                  <input
                    type="text"
                    placeholder="Enter name"
                    value={newPersonName}
                    onChange={(e) => setNewPersonName(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleRegisterFace()}
                  />
                  <button 
                    onClick={handleRegisterFace}
                    disabled={!cameraActive || !newPersonName.trim()}
                  >
                    Register Face
                  </button>
                </div>
                
                <div className="registered-faces">
                  <strong>Registered ({registeredFaces.length}):</strong>
                  {registeredFaces.map(face => (
                    <div key={face.name} className="face-item">
                      <span>{face.name}</span>
                      <span className="face-count">{face.descriptors.length} samples</span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="info">Loading recognition models...</div>
            )}
          </div>

          <div className="stats-panel">
            <h2>Emotion Prediction</h2>
            {prediction && classifierRef.current && classifierRef.current.getTotalSamples() > 0 ? (
              <>
                <div className="prediction">
                  <div className="prediction-label">{prediction.label}</div>
                  <div className="prediction-confidence">
                    {(prediction.confidence * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="probabilities">
                  {EMOTIONS.map(emotion => (
                    <div key={emotion} className="prob-bar">
                      <span>{emotion}</span>
                      <div className="bar">
                        <div 
                          className="bar-fill" 
                          style={{ width: `${prediction.probabilities[emotion] * 100}%` }}
                        />
                      </div>
                      <span>{(prediction.probabilities[emotion] * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="info">Label some expressions to train</div>
            )}
          </div>

          <div className="stats-panel">
            <h2>Label Current Expression</h2>
            <div className="emotion-buttons">
              {EMOTIONS.map(emotion => (
                <button
                  key={emotion}
                  onClick={() => labelExpression(emotion)}
                  disabled={!currentLandmarks}
                  className="emotion-btn"
                >
                  {emotion}
                </button>
              ))}
            </div>
          </div>

          <div className="stats-panel">
            <h2>Training Samples</h2>
            <div className="sample-counts">
              {EMOTIONS.map(emotion => (
                <div key={emotion} className="sample-count">
                  <span>{emotion}:</span>
                  <span>{sampleCounts[emotion]}</span>
                </div>
              ))}
            </div>
            <div className="sample-total">
              Total: {Object.values(sampleCounts).reduce((a, b) => a + b, 0)}
            </div>
          </div>

          <div className="stats-panel">
            <h2>Data Management</h2>
            <div className="data-controls">
              <button onClick={handleExport}>Export Data</button>
              <label className="import-btn">
                Import Data
                <input 
                  type="file" 
                  accept=".json" 
                  onChange={handleImport}
                  style={{ display: 'none' }}
                />
              </label>
              <button onClick={resetLearning} className="danger">
                Reset Learning
              </button>
              <button onClick={resetFaceRecognition} className="danger">
                Reset Face Recognition
              </button>
            </div>
          </div>

          {error && (
            <div className="error-panel">
              <strong>Error:</strong> {error}
              <button onClick={() => setError('')}>×</button>
            </div>
          )}
        </div>
      </div>
      
      {/* Interactive Training Prompt */}
      {showEmotionPrompt && promptedEmotion && (
        <div className="emotion-prompt-overlay">
          <div className="emotion-prompt-modal">
            <h2>How are you feeling right now?</h2>
            <p className="prompt-suggestion">
              The AI detected: <strong>{promptedEmotion}</strong>
            </p>
            <p className="prompt-instruction">
              Please select your actual emotion to improve the model:
            </p>
            
            <div className="prompt-buttons">
              {EMOTIONS.map(emotion => (
                <button
                  key={emotion}
                  onClick={() => handlePromptResponse(emotion)}
                  className={`prompt-emotion-btn ${emotion === promptedEmotion ? 'suggested' : ''}`}
                >
                  <span className="prompt-emoji">
                    {emotion === 'Happy' && '😊'}
                    {emotion === 'Sad' && '😢'}
                    {emotion === 'Angry' && '😠'}
                    {emotion === 'Surprise' && '😲'}
                    {emotion === 'Fear' && '😨'}
                    {emotion === 'Disgust' && '🤢'}
                    {emotion === 'Neutral' && '😐'}
                  </span>
                  <span className="prompt-label">{emotion}</span>
                </button>
              ))}
            </div>
            
            <button 
              onClick={() => handlePromptResponse('none')}
              className="prompt-skip"
            >
              Skip this time
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

