// Debug logging utility for tracking model behavior

export function logPrediction(
  prediction: any,
  features: number[],
  source: string
): void {
  const isDev = import.meta.env.DEV;
  if (isDev) {
    console.log(`[${source}] Prediction:`, {
      label: prediction.label,
      confidence: (prediction.confidence * 100).toFixed(1) + '%',
      featureCount: features.length,
      topProbabilities: Object.entries(prediction.probabilities)
        .sort(([, a], [, b]) => (b as number) - (a as number))
        .slice(0, 3)
        .map(([emotion, prob]) => `${emotion}: ${((prob as number) * 100).toFixed(1)}%`)
    });
  }
}

export function logTraining(
  label: string,
  features: number[],
  sampleCount: number,
  source: string
): void {
  const isDev = import.meta.env.DEV;
  if (isDev) {
    console.log(`[${source}] Training:`, {
      label,
      featureCount: features.length,
      totalSamples: sampleCount,
      timestamp: new Date().toISOString()
    });
  }
}

