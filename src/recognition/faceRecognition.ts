import * as faceapi from '@vladmandic/face-api';
import { FaceDescriptor, RegisteredFace, FaceRecognitionResult, ExpressionResult, EmotionLabel } from '../types';

let modelsLoaded = false;
const RECOGNITION_THRESHOLD = 0.6; // Lower distance = better match
const MAX_DESCRIPTORS_PER_PERSON = 10; // Keep last N descriptors for continual learning

export async function initFaceRecognition(): Promise<void> {
  if (modelsLoaded) return;

  try {
    const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.13/model';
    
    // Load required models including expression detection
    await Promise.all([
      faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
      faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
      faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL)
    ]);

    modelsLoaded = true;
    console.log('Face recognition and expression models loaded successfully');
  } catch (error) {
    console.error('Failed to load face recognition models:', error);
    throw error;
  }
}

export function isRecognitionReady(): boolean {
  return modelsLoaded;
}

// Extract face descriptor from video element
export async function extractFaceDescriptor(
  videoElement: HTMLVideoElement
): Promise<Float32Array | null> {
  if (!modelsLoaded) {
    console.warn('Face recognition models not loaded');
    return null;
  }

  try {
    const detection = await faceapi
      .detectSingleFace(videoElement, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 }))
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (detection && detection.descriptor) {
      return detection.descriptor;
    }

    return null;
  } catch (error) {
    console.error('Error extracting face descriptor:', error);
    return null;
  }
}

// Compute Euclidean distance between two descriptors
function euclideanDistance(desc1: Float32Array, desc2: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < desc1.length; i++) {
    const diff = desc1[i] - desc2[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

// Find best match among registered faces
export function recognizeFace(
  descriptor: Float32Array,
  registeredFaces: RegisteredFace[]
): FaceRecognitionResult | null {
  if (registeredFaces.length === 0) {
    return null;
  }

  let bestMatch: FaceRecognitionResult | null = null;
  let minDistance = Infinity;

  for (const face of registeredFaces) {
    // Compare with all descriptors for this person (continual learning)
    for (const faceDesc of face.descriptors) {
      const distance = euclideanDistance(descriptor, faceDesc.descriptor);
      
      if (distance < minDistance) {
        minDistance = distance;
        bestMatch = {
          name: face.name,
          confidence: Math.max(0, 1 - distance),
          distance: distance
        };
      }
    }
  }

  // Only return match if below threshold
  if (bestMatch && bestMatch.distance < RECOGNITION_THRESHOLD) {
    return bestMatch;
  }

  return null;
}

// Add new descriptor to a person (continual learning)
export function addDescriptorToPerson(
  registeredFaces: RegisteredFace[],
  name: string,
  descriptor: Float32Array
): RegisteredFace[] {
  const updated = [...registeredFaces];
  const personIndex = updated.findIndex(f => f.name === name);

  const newDescriptor: FaceDescriptor = {
    descriptor,
    timestamp: Date.now()
  };

  if (personIndex >= 0) {
    // Person exists - add new descriptor
    const person = { ...updated[personIndex] };
    person.descriptors = [...person.descriptors, newDescriptor];
    
    // Keep only the most recent descriptors (continual learning with bounded memory)
    if (person.descriptors.length > MAX_DESCRIPTORS_PER_PERSON) {
      person.descriptors.sort((a, b) => b.timestamp - a.timestamp);
      person.descriptors = person.descriptors.slice(0, MAX_DESCRIPTORS_PER_PERSON);
    }
    
    person.lastSeen = Date.now();
    person.seenCount++;
    
    updated[personIndex] = person;
  } else {
    // New person - create entry
    updated.push({
      name,
      descriptors: [newDescriptor],
      lastSeen: Date.now(),
      seenCount: 1
    });
  }

  return updated;
}

// Register new person or add sample to existing person
export async function registerFace(
  videoElement: HTMLVideoElement,
  name: string,
  registeredFaces: RegisteredFace[]
): Promise<RegisteredFace[] | null> {
  const descriptor = await extractFaceDescriptor(videoElement);
  
  if (!descriptor) {
    return null;
  }

  return addDescriptorToPerson(registeredFaces, name, descriptor);
}

// Auto-improve: When we recognize someone, add that frame as a new sample (if confident)
export function autoImproveRecognition(
  descriptor: Float32Array,
  recognition: FaceRecognitionResult,
  registeredFaces: RegisteredFace[],
  confidenceThreshold: number = 0.8
): RegisteredFace[] {
  // Only auto-improve if we're very confident
  if (recognition.confidence < confidenceThreshold) {
    return registeredFaces;
  }

  // Add this descriptor as a new sample for continual learning
  return addDescriptorToPerson(registeredFaces, recognition.name, descriptor);
}

// Detect facial expressions using pre-trained model
export async function detectExpressions(
  videoElement: HTMLVideoElement
): Promise<ExpressionResult | null> {
  if (!modelsLoaded) {
    console.warn('Face recognition models not loaded');
    return null;
  }

  try {
    const detection = await faceapi
      .detectSingleFace(videoElement, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 }))
      .withFaceLandmarks()
      .withFaceExpressions();

    if (detection && detection.expressions) {
      const expressions = detection.expressions;
      
      // Map face-api expressions to our emotion labels
      const expressionMap = {
        neutral: expressions.neutral,
        happy: expressions.happy,
        sad: expressions.sad,
        angry: expressions.angry,
        fearful: expressions.fearful,
        disgusted: expressions.disgusted,
        surprised: expressions.surprised
      };

      // Find dominant emotion
      let maxValue = 0;
      let dominantEmotion: EmotionLabel = 'Neutral';
      
      if (expressions.neutral > maxValue) { maxValue = expressions.neutral; dominantEmotion = 'Neutral'; }
      if (expressions.happy > maxValue) { maxValue = expressions.happy; dominantEmotion = 'Happy'; }
      if (expressions.sad > maxValue) { maxValue = expressions.sad; dominantEmotion = 'Sad'; }
      if (expressions.angry > maxValue) { maxValue = expressions.angry; dominantEmotion = 'Angry'; }
      if (expressions.fearful > maxValue) { maxValue = expressions.fearful; dominantEmotion = 'Fear'; }
      if (expressions.disgusted > maxValue) { maxValue = expressions.disgusted; dominantEmotion = 'Disgust'; }
      if (expressions.surprised > maxValue) { maxValue = expressions.surprised; dominantEmotion = 'Surprise'; }

      return {
        expressions: expressionMap,
        dominantEmotion,
        confidence: maxValue
      };
    }

    return null;
  } catch (error) {
    console.error('Error detecting expressions:', error);
    return null;
  }
}

// Merge face recognition descriptor with our geometric features for hybrid model
export function createHybridFeatures(
  geometricFeatures: number[],
  faceDescriptor: Float32Array | null
): number[] {
  if (!faceDescriptor) {
    // If no face descriptor, just use geometric features
    return geometricFeatures;
  }

  // Take first 20 dimensions of face descriptor (128 is too many)
  // and combine with our 11 geometric features
  const compressedDescriptor = Array.from(faceDescriptor.slice(0, 20));
  
  return [...geometricFeatures, ...compressedDescriptor];
}

