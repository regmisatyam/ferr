# Face Expression, Tension, Recognition, and Continual Learning

This is a browser-based application that performs real-time face analysis entirely on your device using your webcam. It detects facial expressions, visualizes facial tension as a heatmap, recognizes registered individuals, and learns incrementally from the labels you provide. An optional AI interview feature lets you explore emotional authenticity by comparing what you say with what your face reveals.

No backend server is required. All face processing happens locally in your browser. The only outbound connection is to the Cerebras API if you choose to run the AI interview.

---

## What the Project Does

The application has four interconnected capabilities:

**Expression Recognition.** The app extracts geometric features from your facial landmarks and uses a small softmax classifier that you train yourself. You make an expression, click the emotion button that matches it, and the classifier updates immediately. The more you label, the more accurate it becomes. You can also optionally combine these hand-crafted features with deep face descriptors from a pre-trained neural network, which often improves accuracy.

**Facial Tension Heatmap.** After you capture a neutral baseline (about three seconds of your relaxed face), the app continuously measures how far each facial landmark has moved from that baseline and renders this as a color-coded heatmap overlaid on your face. The overall deviation is summarized as a tension score from 0 to 100.

**Face Recognition.** You can register your face (or multiple people's faces) with a name. The app then identifies those people in real time using 128-dimensional face embeddings from a pre-trained recognition network. Recognition improves automatically: whenever the system is confident it has recognized someone (above 85% confidence), it saves that frame's descriptor as a new training sample, up to ten descriptors per person.

**AI Interview.** Powered by the Cerebras Cloud API and the Llama 3.1 8B model, this feature conducts a short structured interview. The AI generates five probing questions. For each response, the app records what you said, your detected facial expression, your tension score, and an optional screenshot. If your trained emotion classifier disagrees with the pre-trained expression detector during a response, the system may insert a follow-up question to probe the apparent inconsistency. At the end, the AI analyzes the full session and produces a JSON report covering emotional authenticity, stress indicators, emotional stability, key findings, and recommendations.

---

## Research Context

This project is a practical demonstration of several machine learning ideas applied together in a real-time, client-side system:

**Online continual learning.** The emotion classifier never trains on batches offline. It updates its weights one sample at a time via gradient descent each time you click a label. This is incremental learning in the truest sense.

**Transfer learning.** Rather than training a face recognition network from scratch, the app uses a pre-trained FaceRecognitionNet to produce rich 128-dimensional embeddings. These descriptors are reused both for face identity and (in hybrid mode) as additional features for emotion recognition.

**Feature fusion.** The hybrid model combines 11 hand-crafted geometric features (eye aspect ratios, mouth shape, brow heights, jaw opening, head tilt) with 20 compressed dimensions from the 128-D face descriptor, for a total of 31 features. Combining interpretable geometry with deep representations often outperforms either alone.

**Self-supervised improvement.** Face recognition improves without manual labels: high-confidence detections automatically supply new training samples. This is a form of pseudo-labeling or self-training running continuously in the background.

**Emotion-aware interviewing.** The AI interview demonstrates how a language model can adapt its questioning strategy based on physiological signals. When detected facial emotion conflicts with the trained classifier's prediction, the system infers potential emotional incongruence and probes further. The resulting session report is structured JSON, making it suitable for downstream analysis in research contexts.

---

## Technology Stack

The application is built with React 18 and TypeScript, bundled with Vite 5.

Face landmark detection uses MediaPipe Tasks Vision FaceLandmarker, which returns 468 facial landmarks per frame and runs with GPU acceleration when available. Face recognition and expression detection use the vladmandic fork of face-api.js, which includes SSD face detection, a landmark model, FaceRecognitionNet, and faceExpressionNet. The incremental classifier is implemented from scratch using softmax regression with L2 regularization. All model weights and training data are persisted in IndexedDB so they survive page refreshes. The AI interview uses the official Cerebras Cloud SDK.

---

## Project Structure

```
src/
  vision/
    faceLandmarker.ts       MediaPipe integration and landmark detection
  recognition/
    faceRecognition.ts      Face-api.js integration, recognition logic, hybrid features
  features/
    features.ts             Geometric feature extraction (EAR, MAR, smile, brows, etc.)
  heatmap/
    heatmap.ts              Tension heatmap rendering with jet colormap
    expressionHeatmap.ts    Pre-trained expression visualization
  learning/
    onlineSoftmax.ts        Incremental softmax regression classifier
  storage/
    indexedDb.ts            IndexedDB persistence for all models and data
  ai/
    cerebrasAgent.ts        Cerebras client: question generation, follow-ups, session analysis
  audio/
    speechRecognition.ts    Web Speech API integration for voice input
  charts/
    metricsChart.ts         Session metrics visualization for interview reports
  types.ts                  Shared TypeScript types
  App.tsx                   Main React component and application orchestration
  main.tsx                  Entry point
```

---

## Installation and Setup

**Prerequisites.** Node.js version 18 or higher and a modern browser with WebGL support (Chrome 90+, Edge 90+, Firefox 88+, or Safari 15+). Camera access requires either HTTPS or localhost.

**Install dependencies.**

```bash
npm install
```

**Start the development server.**

```bash
npm run dev
```

The app opens at `http://localhost:5173` by default.

**Build for production.**

```bash
npm run build
```

**Preview the production build.**

```bash
npm run preview
```

---

## How to Use

**Step 1: Start the camera.** Click the "Start Camera" button and allow camera permissions. You should see your face with landmark points detected.

**Step 2: Capture a neutral baseline.** Relax your face into a neutral expression and click "Capture Neutral Baseline." Hold still for about three seconds while 30 frames are collected. Once captured, the tension heatmap becomes active.

**Step 3: Register your face (optional).** Type your name into the input field and click "Register Face." You can register multiple people by entering different names. The system will recognize them in subsequent sessions.

**Step 4: Enable hybrid mode (optional).** Check the "Hybrid Model (Geo + Face)" checkbox to combine geometric features with deep face descriptors. This can improve emotion classification accuracy, especially when expressions vary between individuals. It is slightly slower.

**Step 5: Train the emotion classifier.** Make an expression and click the corresponding emotion button (Neutral, Happy, Angry, Sad, Surprise, Fear, Disgust). Repeat for different expressions. The model updates after every single sample. Aim for at least five to ten samples per emotion for reliable predictions.

**Step 6: Run the AI interview (optional).** Click "Start AI Interview." You will be prompted to enter a Cerebras API key, which you can obtain for free at cloud.cerebras.ai. The AI will ask five questions. You can answer by typing or by using the voice input button if your browser supports it. Press Ctrl+Enter to submit each response. After the final question, the AI generates a full session report that you can download as JSON.

**Step 7: Export or import your data.** Use "Export Data" to save all training samples, model weights, baseline, and registered faces as a JSON file. Use "Import Data" to restore a previous export. Data is also automatically saved in the browser's IndexedDB, so it persists between sessions without any manual action.

---

## Algorithm Details

**Geometric features.** Eleven features are extracted per frame: left and right eye aspect ratio (EAR), mouth aspect ratio (MAR), smile curvature, left and right eyebrow heights, inter-ocular distance, eye openness asymmetry, mouth width, jaw opening, and head tilt angle. All features are normalized by inter-ocular distance to be scale-invariant.

EAR formula: (distance between vertical eye landmarks top-to-bottom on each side, summed) divided by (twice the horizontal eye width).

MAR formula: vertical mouth opening divided by horizontal mouth width.

**Tension computation.** For each of the 468 landmarks, the app computes the Euclidean distance between the current position and the baseline position captured during the neutral baseline step. These distances are normalized to the range 0 to 1 using the maximum deviation in the current frame. The result is rendered as a colored heatmap using the jet colormap (blue for low tension, green and yellow for moderate, red for high). The overall tension score is the mean normalized tension multiplied by 100.

**Face recognition.** FaceRecognitionNet produces a 128-dimensional embedding for each detected face. Matching uses Euclidean distance with a threshold of 0.6. Up to ten descriptors are stored per person, keeping the most recent ones. Any detection above 85% confidence automatically appends a new descriptor sample.

**Hybrid feature fusion.** The 128-D face descriptor is compressed to 20 dimensions by taking the first 20 values. These are concatenated with the 11 geometric features to form a 31-dimensional feature vector. The softmax classifier resizes its weight matrix automatically when you switch between geometric-only (11-D) and hybrid (31-D) modes.

**Online softmax classifier.** This is multinomial logistic regression with immediate weight updates. For each labeled sample, the app computes the cross-entropy gradient and applies a single gradient descent step with learning rate 0.05 and L2 regularization coefficient 0.001. The weight update rule is: W = W minus learning-rate times (gradient times features plus lambda times W). There is no batch accumulation; the model updates on every single labeled sample.

---

## AI Interview: How It Works

When you start an interview, the Cerebras agent generates five probing questions asking you to reflect on emotions, stress, and your current state. The questions are a mix of open-ended, scale, and yes/no formats.

During each response the app records your answer text, the dominant expression detected by the pre-trained expression network, the confidence of that detection, and your facial tension score at the time you submitted.

After you submit a response, the system compares the prediction from your user-trained classifier against the pre-trained expression reading. If they disagree and there is room in the question queue, the agent generates a follow-up question that acknowledges the apparent inconsistency and invites you to reflect on it.

After all questions are answered, the agent receives the full transcript together with the sequence of detected emotions and generates a structured analysis. This analysis includes a dominant emotion, an emotional stability score, a stress level, an authenticity score (reflecting agreement between verbal content and facial expressions), key findings, and recommendations. If the API call fails for any reason, a local heuristic based on emotion counts provides a fallback report.

The session report is available for download as a JSON file. A visual metrics chart is also generated and can be downloaded separately.

---

## Privacy

All face processing happens locally in your browser. Your webcam feed is never sent anywhere. Training data, model weights, baseline frames, and face descriptors are stored in your browser's IndexedDB and never leave your device unless you explicitly export them.

The only exception is the AI interview feature, which sends your text responses and detected emotion metadata to the Cerebras API for analysis. No images or raw video are sent; the screenshot embedded in the report is stored locally in the downloaded JSON file only.

---

## Performance Notes

The app targets 30 frames per second. Actual FPS is displayed in the stats panel. GPU acceleration is enabled for MediaPipe when the browser supports it. The heatmap blur and face descriptor extraction are the most computationally intensive operations. On slower devices, you can disable the heatmap, turn off hybrid mode, or close other browser tabs to improve performance.

---

## Troubleshooting

**Camera not working.** Make sure you are on localhost or an HTTPS URL. Check that the browser has camera permission. Try a different browser if needed.

**No face detected.** Improve lighting, move closer to the camera, face the camera directly, and avoid anything covering your face.

**Low FPS.** Close other applications, disable debug mode, or switch from hybrid to geometric-only mode.

**Classifier not improving.** Collect at least five to ten samples per emotion with clearly distinct expressions. Make sure a face is actively detected when you click the label buttons. Enabling hybrid mode can help if geometric features alone are not separating emotions well.

**Face recognition not working.** Register your face in good lighting facing the camera directly. Recognition accuracy improves automatically over multiple sessions as the system accumulates high-confidence observations.

**AI interview fails to initialize.** Verify that your Cerebras API key is correct and that your account has remaining credits. Make sure you have an active internet connection.

**Voice input not available.** Voice input relies on the Web Speech API, which is supported in Chrome, Edge, and Safari. Firefox does not support it. Type responses manually if voice input is unavailable.

---

## Credits

Face landmark detection is powered by MediaPipe Tasks Vision from Google. Face recognition and expression detection use the vladmandic/face-api fork, which is an optimized version of face-api.js with the FaceRecognitionNet architecture originally from the face-recognition.js project. Facial landmark indices follow the MediaPipe Face Mesh topology. The AI interview runs on the Cerebras Cloud platform using the Llama 3.1 8B model.

---

## License

MIT License. Use and modify freely.
