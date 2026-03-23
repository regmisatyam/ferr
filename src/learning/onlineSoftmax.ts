import { EmotionLabel, ClassifierState, Prediction } from '../types';
import { saveClassifierState, loadClassifierState } from '../storage/indexedDb';

const EMOTION_LABELS: EmotionLabel[] = ['Neutral', 'Happy', 'Angry', 'Sad', 'Surprise', 'Fear', 'Disgust'];
const LEARNING_RATE = 0.05;
const L2_LAMBDA = 0.001; // L2 regularization

export class OnlineSoftmaxClassifier {
  private weights: number[][];
  private bias: number[];
  private sampleCounts: Record<EmotionLabel, number>;
  private numFeatures: number;
  private numClasses: number;

  constructor(numFeatures: number, _useHybridFeatures: boolean = false) {
    this.numFeatures = numFeatures;
    this.numClasses = EMOTION_LABELS.length;
    
    // Initialize weights and bias to small random values
    this.weights = Array(this.numClasses)
      .fill(0)
      .map(() => Array(numFeatures).fill(0).map(() => (Math.random() - 0.5) * 0.01));
    
    this.bias = Array(this.numClasses).fill(0);
    
    this.sampleCounts = {
      Neutral: 0,
      Happy: 0,
      Angry: 0,
      Sad: 0,
      Surprise: 0,
      Fear: 0,
      Disgust: 0
    };
  }

  // Softmax function
  private softmax(logits: number[]): number[] {
    const maxLogit = Math.max(...logits);
    const exps = logits.map(l => Math.exp(l - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sumExps);
  }

  // Forward pass: compute probabilities
  predict(features: number[]): Prediction {
    // Ensure feature dimensions match
    if (features.length !== this.numFeatures) {
      console.warn(`Feature dimension mismatch: expected ${this.numFeatures}, got ${features.length}`);
      // Pad or truncate features if needed
      if (features.length < this.numFeatures) {
        features = [...features, ...Array(this.numFeatures - features.length).fill(0)];
      } else {
        features = features.slice(0, this.numFeatures);
      }
    }
    
    // Compute logits: W * x + b
    const logits = this.weights.map((w, i) => {
      const dotProduct = w.reduce((sum, weight, j) => sum + weight * features[j], 0);
      return dotProduct + this.bias[i];
    });

    // Apply softmax
    const probabilities = this.softmax(logits);

    // Find max probability
    let maxProb = 0;
    let maxIdx = 0;
    for (let i = 0; i < probabilities.length; i++) {
      if (probabilities[i] > maxProb) {
        maxProb = probabilities[i];
        maxIdx = i;
      }
    }

    const probabilitiesMap: Record<EmotionLabel, number> = {} as Record<EmotionLabel, number>;
    EMOTION_LABELS.forEach((label, i) => {
      probabilitiesMap[label] = probabilities[i];
    });

    return {
      label: EMOTION_LABELS[maxIdx],
      confidence: maxProb,
      probabilities: probabilitiesMap
    };
  }

  // Online learning: update weights with one sample
  async train(features: number[], label: EmotionLabel): Promise<void> {
    // Get current prediction
    const prediction = this.predict(features);
    const probs = Object.values(prediction.probabilities);

    // Create one-hot target
    const target = EMOTION_LABELS.map(l => (l === label ? 1 : 0));

    // Compute gradient: (pred - target)
    const gradient = probs.map((p, i) => p - target[i]);

    // Update weights with gradient descent and L2 regularization
    for (let i = 0; i < this.numClasses; i++) {
      for (let j = 0; j < this.numFeatures; j++) {
        const regularization = L2_LAMBDA * this.weights[i][j];
        this.weights[i][j] -= LEARNING_RATE * (gradient[i] * features[j] + regularization);
      }
      this.bias[i] -= LEARNING_RATE * gradient[i];
    }

    // Update sample count
    this.sampleCounts[label]++;

    // Persist to IndexedDB
    await this.save();
  }

  // Save state to IndexedDB
  async save(): Promise<void> {
    const state: ClassifierState = {
      weights: this.weights,
      bias: this.bias,
      sampleCounts: this.sampleCounts
    };
    await saveClassifierState(state);
  }

  // Load state from IndexedDB
  async load(): Promise<boolean> {
    const state = await loadClassifierState();
    if (state) {
      this.weights = state.weights;
      this.bias = state.bias;
      this.sampleCounts = state.sampleCounts;
      
      // Update numFeatures if loaded model has different size (hybrid vs non-hybrid)
      if (this.weights.length > 0 && this.weights[0].length !== this.numFeatures) {
        this.numFeatures = this.weights[0].length;
      }
      
      return true;
    }
    return false;
  }
  
  // Resize model if switching between hybrid and non-hybrid features
  resizeForHybrid(newNumFeatures: number): void {
    if (newNumFeatures === this.numFeatures) return;
    
    // Expand or shrink weights
    const oldWeights = this.weights;
    this.weights = Array(this.numClasses)
      .fill(0)
      .map((_, i) => {
        const newRow = Array(newNumFeatures).fill(0).map(() => (Math.random() - 0.5) * 0.01);
        // Copy existing weights if available
        if (oldWeights[i]) {
          const copyLen = Math.min(oldWeights[i].length, newNumFeatures);
          for (let j = 0; j < copyLen; j++) {
            newRow[j] = oldWeights[i][j];
          }
        }
        return newRow;
      });
    
    this.numFeatures = newNumFeatures;
  }

  // Reset classifier
  reset(): void {
    this.weights = Array(this.numClasses)
      .fill(0)
      .map(() => Array(this.numFeatures).fill(0).map(() => (Math.random() - 0.5) * 0.01));
    
    this.bias = Array(this.numClasses).fill(0);
    
    this.sampleCounts = {
      Neutral: 0,
      Happy: 0,
      Angry: 0,
      Sad: 0,
      Surprise: 0,
      Fear: 0,
      Disgust: 0
    };
  }

  getSampleCounts(): Record<EmotionLabel, number> {
    return { ...this.sampleCounts };
  }

  getTotalSamples(): number {
    return Object.values(this.sampleCounts).reduce((a, b) => a + b, 0);
  }
}

