import { Landmark, FeatureVector } from '../types';

// FaceMesh landmark indices
const LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380];
const MOUTH_LEFT = 61;
const MOUTH_RIGHT = 291;
const MOUTH_UPPER = 13;
const MOUTH_LOWER = 14;
const LEFT_EYE_CENTER = 33;
const RIGHT_EYE_CENTER = 263;
const LEFT_BROW = 70;
const RIGHT_BROW = 300;
const CHIN = 152;
const NOSE_TIP = 0;

function euclideanDistance2D(p1: Landmark, p2: Landmark): number {
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function computeEAR(eyeIndices: number[], landmarks: Landmark[]): number {
  // EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
  const p1 = landmarks[eyeIndices[0]];
  const p2 = landmarks[eyeIndices[1]];
  const p3 = landmarks[eyeIndices[2]];
  const p4 = landmarks[eyeIndices[3]];
  const p5 = landmarks[eyeIndices[4]];
  const p6 = landmarks[eyeIndices[5]];

  const vertical1 = euclideanDistance2D(p2, p6);
  const vertical2 = euclideanDistance2D(p3, p5);
  const horizontal = euclideanDistance2D(p1, p4);

  if (horizontal < 1e-6) return 0;
  return (vertical1 + vertical2) / (2.0 * horizontal);
}

export function extractFeatures(landmarks: Landmark[]): FeatureVector {
  // Ensure we have enough landmarks
  if (landmarks.length < 478) {
    console.warn(`Expected 478 landmarks, got ${landmarks.length}`);
  }

  // 1. Left and Right EAR
  const leftEAR = computeEAR(LEFT_EYE_INDICES, landmarks);
  const rightEAR = computeEAR(RIGHT_EYE_INDICES, landmarks);

  // 2. MAR (Mouth Aspect Ratio)
  const mouthCornerLeft = landmarks[MOUTH_LEFT];
  const mouthCornerRight = landmarks[MOUTH_RIGHT];
  const mouthUpper = landmarks[MOUTH_UPPER];
  const mouthLower = landmarks[MOUTH_LOWER];
  
  const mouthHeight = euclideanDistance2D(mouthUpper, mouthLower);
  const mouthWidthDist = euclideanDistance2D(mouthCornerLeft, mouthCornerRight);
  const mar = mouthWidthDist > 1e-6 ? mouthHeight / mouthWidthDist : 0;

  // 3. Smile curvature (distance from mid-corners to upper lip)
  const midCornerX = (mouthCornerLeft.x + mouthCornerRight.x) / 2;
  const midCornerY = (mouthCornerLeft.y + mouthCornerRight.y) / 2;
  const midCorner = { x: midCornerX, y: midCornerY, z: 0 };
  const smileCurvature = euclideanDistance2D(midCorner, mouthUpper);

  // 4. Eyebrow heights
  const leftEyeCenter = landmarks[LEFT_EYE_CENTER];
  const rightEyeCenter = landmarks[RIGHT_EYE_CENTER];
  const leftBrowPoint = landmarks[LEFT_BROW];
  const rightBrowPoint = landmarks[RIGHT_BROW];
  
  const leftBrowHeight = euclideanDistance2D(leftBrowPoint, leftEyeCenter);
  const rightBrowHeight = euclideanDistance2D(rightBrowPoint, rightEyeCenter);

  // 5. Inter-ocular distance (for normalization)
  const interOcularDistance = euclideanDistance2D(leftEyeCenter, rightEyeCenter);

  // 6. Eye openness asymmetry
  const eyeOpennessAsymmetry = leftEAR - rightEAR;

  // 7. Mouth width
  const mouthWidth = mouthWidthDist;

  // 8. Jaw opening
  const chin = landmarks[CHIN];
  const noseTip = landmarks[NOSE_TIP];
  const jawOpening = euclideanDistance2D(chin, noseTip);

  // 9. Head tilt angle (from eye line)
  const dx = rightEyeCenter.x - leftEyeCenter.x;
  const dy = rightEyeCenter.y - leftEyeCenter.y;
  const headTiltAngle = Math.atan2(dy, dx);

  return {
    leftEAR,
    rightEAR,
    mar,
    smileCurvature,
    leftBrowHeight,
    rightBrowHeight,
    interOcularDistance,
    eyeOpennessAsymmetry,
    mouthWidth,
    jawOpening,
    headTiltAngle
  };
}

export function featuresToArray(features: FeatureVector): number[] {
  return [
    features.leftEAR,
    features.rightEAR,
    features.mar,
    features.smileCurvature,
    features.leftBrowHeight,
    features.rightBrowHeight,
    features.interOcularDistance,
    features.eyeOpennessAsymmetry,
    features.mouthWidth,
    features.jawOpening,
    features.headTiltAngle
  ];
}

export function normalizeFeatures(features: number[]): number[] {
  // Normalize by inter-ocular distance (index 6)
  const iod = features[6];
  if (iod < 1e-6) return features;

  return features.map((f, idx) => {
    // Don't normalize the IOD itself, asymmetry, or angle
    if (idx === 6 || idx === 7 || idx === 10) return f;
    return f / iod;
  });
}

