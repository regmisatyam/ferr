import { FaceLandmarker, FilesetResolver, FaceLandmarkerResult } from '@mediapipe/tasks-vision';
import { Landmark } from '../types';

let faceLandmarker: FaceLandmarker | null = null;

export async function initFaceLandmarker(): Promise<void> {
  try {
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm'
    );

    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate: 'GPU'
      },
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
      runningMode: 'VIDEO',
      numFaces: 1
    });

    console.log('FaceLandmarker initialized successfully');
  } catch (error) {
    console.error('Failed to initialize FaceLandmarker:', error);
    throw error;
  }
}

export function detectLandmarks(
  videoElement: HTMLVideoElement,
  timestamp: number
): Landmark[] | null {
  if (!faceLandmarker) {
    console.warn('FaceLandmarker not initialized');
    return null;
  }

  try {
    const results: FaceLandmarkerResult = faceLandmarker.detectForVideo(videoElement, timestamp);

    if (results.faceLandmarks && results.faceLandmarks.length > 0) {
      // Get the first face's landmarks
      const landmarks = results.faceLandmarks[0];
      
      // Convert to our Landmark format
      return landmarks.map(lm => ({
        x: lm.x,
        y: lm.y,
        z: lm.z || 0
      }));
    }

    return null;
  } catch (error) {
    console.error('Error detecting landmarks:', error);
    return null;
  }
}

export function isInitialized(): boolean {
  return faceLandmarker !== null;
}

