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

export interface AgenticQuestion {
  id: string;
  question: string;
  type: 'open' | 'yesno' | 'scale' | 'choice';
  options?: string[];
  followUp?: boolean;
}

export interface UserResponse {
  questionId: string;
  question: string;
  response: string;
  timestamp: number;
  emotion: EmotionLabel;
  emotionConfidence: number;
  facialTension: number;
  imageData?: string;
  audioTranscript?: string;
}

export interface InteractionReport {
  sessionId: string;
  startTime: number;
  endTime: number;
  responses: UserResponse[];
  emotionalJourney: {
    timestamp: number;
    emotion: EmotionLabel;
    confidence: number;
  }[];
  insights: {
    dominantEmotion: EmotionLabel;
    emotionalStability: number;
    stressLevel: number;
    authenticity: number;
    keyFindings: string[];
    recommendations: string[];
  };
  summary: string;
}

export interface AgentSession {
  id: string;
  active: boolean;
  questionIndex: number;
  questions: AgenticQuestion[];
  responses: UserResponse[];
  startTime: number;
}
