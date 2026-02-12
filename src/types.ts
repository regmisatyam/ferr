export interface Landmark {
  x: number;
  y: number;
  z: number;
}

export interface FeatureVector {
  leftEAR: number;
  rightEAR: number;
  mar: number;
  smileCurvature: number;
  leftBrowHeight: number;
  rightBrowHeight: number;
  interOcularDistance: number;
  eyeOpennessAsymmetry: number;
  mouthWidth: number;
  jawOpening: number;
  headTiltAngle: number;
}

export type EmotionLabel = 'Neutral' | 'Happy' | 'Angry' | 'Sad' | 'Surprise' | 'Fear' | 'Disgust';

export interface TrainingSample {
  timestamp: number;
  features: number[];
  label: EmotionLabel;
}

export interface ClassifierState {
  weights: number[][];
  bias: number[];
  sampleCounts: Record<EmotionLabel, number>;
}

export interface Prediction {
  label: EmotionLabel;
  confidence: number;
  probabilities: Record<EmotionLabel, number>;
}

export interface FaceDescriptor {
  descriptor: Float32Array;
  timestamp: number;
}

export interface RegisteredFace {
  name: string;
  descriptors: FaceDescriptor[];
  lastSeen: number;
  seenCount: number;
}

export interface FaceRecognitionResult {
  name: string;
  confidence: number;
  distance: number;
}

export interface FaceExpressions {
  neutral: number;
  happy: number;
  sad: number;
  angry: number;
  fearful: number;
  disgusted: number;
  surprised: number;
}

export interface ExpressionResult {
  expressions: FaceExpressions;
  dominantEmotion: EmotionLabel;
  confidence: number;
}

