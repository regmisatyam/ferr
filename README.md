# Face Expression + Tension Heatmap + Face Recognition + Continual Learning

A comprehensive client-side browser application that performs real-time face expression analysis, facial tension visualization, face recognition, and continual learning using webcam input. Features both geometric and deep learning-based face analysis with hybrid model capabilities.

## Features

### 1. **Webcam Face Detection**
- Uses MediaPipe Tasks Vision FaceLandmarker for detecting 468/478 facial landmarks
- Real-time video processing at 30 FPS
- No backend required - runs entirely in the browser

### 2. **Geometric Feature Extraction**
The app computes 11 numeric features from facial landmarks:
- **Left & Right EAR (Eye Aspect Ratio)**: Measures eye openness
- **MAR (Mouth Aspect Ratio)**: Measures mouth opening
- **Smile Curvature**: Detects smile intensity
- **Eyebrow Heights**: Left and right eyebrow elevation
- **Inter-Ocular Distance**: Distance between eyes (for normalization)
- **Eye Openness Asymmetry**: Difference between left and right eye openness
- **Mouth Width**: Horizontal mouth dimension
- **Jaw Opening**: Vertical jaw displacement
- **Head Tilt Angle**: Head rotation angle

### 3. **Baseline Capture & Tension Heatmap**
- **Capture Baseline**: Press "Capture Neutral Baseline" to collect 30 frames of your neutral expression
- **Tension Visualization**: Real-time heatmap overlay showing per-landmark tension (deviation from baseline)
- **Tension Score**: 0-100 metric indicating overall facial tension
- **Colormap**: Jet colormap (blue → green → yellow → red) for intuitive visualization

### 4. **Continual Learning**
- **Label Expressions**: Click emotion buttons (Neutral, Happy, Angry, Sad, Surprise, Fear, Disgust) to label your current expression
- **Incremental Classifier**: Online softmax regression that updates immediately with each labeled sample
- **Persistent Learning**: Model weights and training data stored in IndexedDB, survives page refreshes
- **Real-time Prediction**: Shows predicted emotion with confidence scores and probability distribution

### 5. **Face Recognition with Continual Learning**
- **Face Registration**: Register faces with names for identification
- **Real-time Recognition**: Identifies registered individuals with confidence scores
- **Auto-Improvement**: Automatically adds new samples when recognition is confident (>85%)
- **Multi-Sample Learning**: Stores up to 10 descriptors per person for robust recognition
- **128-D Face Descriptors**: Uses face-api.js with FaceRecognitionNet for deep face embeddings
- **Persistent Storage**: All registered faces saved in IndexedDB

### 6. **Hybrid Model Architecture**
- **Geometric Features**: 11 hand-crafted facial geometry features (EAR, MAR, etc.)
- **Deep Features**: 128-dimensional face recognition descriptors (compressed to 20-D)
- **Hybrid Mode**: Combines both feature types for improved emotion recognition
- **Dynamic Switching**: Toggle between geometric-only and hybrid models
- **Model Adaptation**: Automatically resizes classifier when switching modes

### 7. **Data Management**
- **Export Data**: Download all training data, baseline, registered faces, and model weights as JSON
- **Import Data**: Load previously exported data to continue learning
- **Reset Learning**: Clear all emotion training data and reset the model
- **Reset Face Recognition**: Delete all registered faces

### 8. **Debug Mode**
- Toggle to visualize landmark points and indices for verification
- Useful for understanding which facial points are being tracked

## Installation & Setup

### Prerequisites
- Node.js (v18 or higher)
- Modern web browser with WebGL support

### Install Dependencies
```bash
npm install
```

### Run Development Server
```bash
npm run dev
```

The app will open at `http://localhost:5173` (or similar).

### Build for Production
```bash
npm run build
```

### Preview Production Build
```bash
npm run preview
```

## Usage Guide

### Step 1: Start Camera
1. Click "Start Camera" button
2. Allow camera permissions when prompted
3. Position your face in view (you should see landmarks detected)

### Step 2: Capture Baseline
1. Make a neutral, relaxed expression
2. Click "Capture Neutral Baseline"
3. Hold still for ~3 seconds while 30 frames are collected
4. The tension heatmap will now activate

### Step 3: Register Your Face (Optional)
1. Enter your name in the input field
2. Click "Register Face" button
3. Your face is now registered for recognition
4. Register multiple people by entering different names
5. The system learns continually - recognition improves over time

### Step 4: Enable Hybrid Model (Optional)
1. Check "Hybrid Model (Geo + Face)" checkbox
2. This combines geometric features with face recognition descriptors
3. Can improve emotion classification accuracy
4. Slightly slower but more powerful

### Step 5: Train the Emotion Classifier
1. Make an expression (e.g., smile for "Happy")
2. Click the corresponding emotion button
3. Repeat for various expressions and emotions
4. The model updates immediately with each label
5. Watch as predictions improve with more training samples

### Step 6: Monitor Results
- **Face Recognition**: Shows identified person with confidence
- **Tension Score**: Shows how much your face deviates from baseline
- **Emotion Prediction**: Displays predicted emotion with confidence
- **Training Samples**: Shows count of labeled samples per emotion

### Step 7: Export/Import Data (Optional)
- Use "Export Data" to save your training progress and registered faces
- Use "Import Data" to restore from a previous export
- Data is also automatically persisted in browser storage

## Technical Architecture

### Project Structure
```
/src
  /vision
    faceLandmarker.ts      # MediaPipe FaceLandmarker integration (468 landmarks)
  /recognition
    faceRecognition.ts     # Face-api.js integration & recognition logic
  /features
    features.ts            # Geometric feature extraction (EAR, MAR, etc.)
  /heatmap
    heatmap.ts             # Heatmap rendering & Jet colormap
  /learning
    onlineSoftmax.ts       # Incremental softmax regression classifier
  /storage
    indexedDb.ts           # IndexedDB persistence layer
  types.ts                 # TypeScript interfaces
  App.tsx                  # Main React component with full UI
  App.css                  # Styling
  main.tsx                 # Entry point
```

### Key Technologies
- **Vite**: Fast build tool and dev server
- **React + TypeScript**: UI framework with type safety
- **MediaPipe Tasks Vision**: 468-point face landmark detection
- **face-api.js (@vladmandic/face-api)**: Face recognition with FaceRecognitionNet
- **IndexedDB**: Client-side data persistence
- **Canvas API**: Video processing and heatmap rendering

### Algorithm Details

#### Feature Extraction
- **EAR Formula**: `(||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)`
  - Uses specific eye landmark indices for vertical/horizontal distances
- **MAR Formula**: `||mouth_top-mouth_bottom|| / ||mouth_left-mouth_right||`
- Features are normalized by inter-ocular distance for scale invariance

#### Tension Computation
1. Compute Euclidean distance between each current landmark and baseline landmark (in pixel space)
2. Normalize distances to [0,1] using max tension in current frame
3. Render as colored circles with blur for smooth heatmap
4. Overall tension score = mean of normalized tensions × 100

#### Face Recognition
- **Algorithm**: FaceRecognitionNet (128-D descriptors)
- **Matching**: Euclidean distance between descriptors
- **Threshold**: 0.6 distance for positive match
- **Continual Learning**: Stores multiple descriptors per person
- **Auto-Improvement**: High-confidence recognitions (>85%) automatically add new samples
- **Bounded Memory**: Keeps most recent 10 descriptors per person

#### Hybrid Feature Fusion
- **Geometric**: 11 features (EAR, MAR, angles, distances)
- **Deep**: 20 compressed face descriptor dimensions (from 128-D)
- **Total**: 31 hybrid features when enabled
- **Adaptive**: Model weights resize automatically when switching modes

#### Online Learning
- **Algorithm**: Softmax regression (multinomial logistic regression)
- **Update Rule**: Gradient descent with cross-entropy loss
- **Learning Rate**: 0.05
- **Regularization**: L2 with λ=0.001
- **Weight Update**: `W -= lr * (gradient * features + λ * W)`
- **Feature Flexibility**: Supports both 11-D geometric and 31-D hybrid features
- Each labeled sample immediately updates weights via one-step gradient descent

## Browser Compatibility
- Chrome 90+ (recommended)
- Edge 90+
- Firefox 88+
- Safari 15+

**Note**: HTTPS or localhost required for camera access.

## Performance Notes
- Target: 30 FPS
- Actual FPS displayed in stats panel
- GPU acceleration enabled for MediaPipe when available
- Heatmap blur may impact performance on slower devices

## Troubleshooting

### Camera Not Working
- Ensure HTTPS or localhost
- Check browser permissions
- Try different browser if issues persist

### "No Face Detected"
- Ensure good lighting
- Move closer to camera
- Face camera directly
- Avoid occlusions (hats, hands, etc.)

### Low FPS
- Close other tabs/applications
- Reduce browser window size
- Disable debug mode
- Use newer hardware with GPU support

### Model Not Learning
- Ensure diverse expressions for each emotion
- Collect at least 5-10 samples per emotion
- Verify face is detected when labeling
- Try enabling hybrid model for better accuracy

### Face Recognition Not Working
- Ensure good lighting and face camera directly
- Register at least 1-2 samples per person
- Recognition improves automatically with more exposures
- Check that face detection is working (green border or debug mode)

## Data Privacy
- All processing happens client-side
- No data sent to servers
- Webcam feed never leaves your browser
- Training data stored locally in IndexedDB
- Export/import for manual data portability

## License
MIT License - Feel free to use and modify as needed.

## Advanced Features

### Continual Learning Explained
This app implements true continual learning in two ways:

1. **Emotion Recognition**: Each time you label an expression, the softmax classifier updates its weights immediately via gradient descent. No batch training needed - the model learns incrementally.

2. **Face Recognition**: When the system recognizes you with high confidence (>85%), it automatically adds that frame's face descriptor as a new training sample. Over time, recognition accuracy improves without manual intervention.

### Why Hybrid Models?
- **Geometric features** are interpretable, fast, and work well for expressions
- **Deep features** capture identity, subtle patterns, and texture information
- **Combining both** often yields better emotion recognition, especially when expressions vary by person
- Trade-off: Hybrid mode is slightly slower due to extra descriptor extraction

### Model Persistence
- All models and data stored in browser's IndexedDB
- Survives page refreshes and browser restarts
- Export/import for backup or transfer between devices
- Each person's face gets multiple descriptors for robustness

## Credits
- Built with [MediaPipe](https://developers.google.com/mediapipe) for 468-point face landmark detection
- Face recognition powered by [@vladmandic/face-api](https://github.com/vladmandic/face-api) (optimized fork)
- Facial action unit indices based on [Face Mesh](https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md)
- FaceRecognitionNet architecture from [face-recognition.js](https://github.com/justadudewhohacks/face-recognition.js)

## Research Applications
This app demonstrates several ML concepts:
- **Online learning** (continual/incremental learning)
- **Transfer learning** (using pre-trained face recognition)
- **Feature fusion** (geometric + deep features)
- **Multi-task learning** (identity + expression)
- **Self-supervised improvement** (auto-labeling with confidence threshold)

---

**Enjoy exploring facial expressions, recognition, and continual learning! 🎭📊🧠👤**

