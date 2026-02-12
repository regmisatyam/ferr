import { Landmark } from '../types';

// Jet colormap: maps 0..1 to RGB
export function jetColormap(value: number): [number, number, number] {
  // Clamp value to [0, 1]
  const v = Math.max(0, Math.min(1, value));

  let r, g, b;

  if (v < 0.25) {
    r = 0;
    g = 4 * v;
    b = 1;
  } else if (v < 0.5) {
    r = 0;
    g = 1;
    b = 1 - 4 * (v - 0.25);
  } else if (v < 0.75) {
    r = 4 * (v - 0.5);
    g = 1;
    b = 0;
  } else {
    r = 1;
    g = 1 - 4 * (v - 0.75);
    b = 0;
  }

  return [
    Math.round(r * 255),
    Math.round(g * 255),
    Math.round(b * 255)
  ];
}

export function computeTension(
  currentLandmarks: Landmark[],
  baselineLandmarks: Landmark[],
  videoWidth: number,
  videoHeight: number
): { tensions: number[]; tensionScore: number } {
  if (currentLandmarks.length !== baselineLandmarks.length) {
    return { tensions: [], tensionScore: 0 };
  }

  const tensions: number[] = [];
  
  // Compute per-landmark tension (Euclidean distance in pixel space)
  for (let i = 0; i < currentLandmarks.length; i++) {
    const curr = currentLandmarks[i];
    const base = baselineLandmarks[i];

    const currX = curr.x * videoWidth;
    const currY = curr.y * videoHeight;
    const baseX = base.x * videoWidth;
    const baseY = base.y * videoHeight;

    const dx = currX - baseX;
    const dy = currY - baseY;
    const distance = Math.sqrt(dx * dx + dy * dy);

    tensions.push(distance);
  }

  // Normalize tensions to [0, 1]
  const maxTension = Math.max(...tensions, 1e-6);
  const normalizedTensions = tensions.map(t => t / maxTension);

  // Compute overall tension score (mean of normalized tensions)
  const tensionScore = normalizedTensions.reduce((a, b) => a + b, 0) / normalizedTensions.length;

  return {
    tensions: normalizedTensions,
    tensionScore: Math.min(100, tensionScore * 100)
  };
}

export function renderHeatmap(
  canvas: HTMLCanvasElement,
  videoElement: HTMLVideoElement,
  currentLandmarks: Landmark[],
  baselineLandmarks: Landmark[] | null,
  showDebug: boolean
): number {
  const ctx = canvas.getContext('2d');
  if (!ctx) return 0;

  const width = videoElement.videoWidth;
  const height = videoElement.videoHeight;

  if (width === 0 || height === 0) return 0;

  // Set canvas size to match video
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  // Draw video frame
  ctx.drawImage(videoElement, 0, 0, width, height);

  // If no baseline, just show debug landmarks if enabled
  if (!baselineLandmarks) {
    if (showDebug) {
      drawDebugLandmarks(ctx, currentLandmarks, width, height);
    }
    return 0;
  }

  // Compute tensions
  const { tensions, tensionScore } = computeTension(
    currentLandmarks,
    baselineLandmarks,
    width,
    height
  );

  // Create heatmap overlay
  const heatmapCanvas = document.createElement('canvas');
  heatmapCanvas.width = width;
  heatmapCanvas.height = height;
  const heatmapCtx = heatmapCanvas.getContext('2d');
  
  if (!heatmapCtx) return tensionScore;

  // Draw tension circles for each landmark
  for (let i = 0; i < currentLandmarks.length; i++) {
    const landmark = currentLandmarks[i];
    const tension = tensions[i];

    const x = landmark.x * width;
    const y = landmark.y * height;

    // Draw circle with intensity based on tension
    const color = jetColormap(tension);
    const alpha = tension * 0.7; // Scale alpha by tension

    heatmapCtx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${alpha})`;
    heatmapCtx.beginPath();
    heatmapCtx.arc(x, y, 3, 0, 2 * Math.PI);
    heatmapCtx.fill();
  }

  // Apply blur for smoother heatmap
  heatmapCtx.filter = 'blur(8px)';
  heatmapCtx.drawImage(heatmapCanvas, 0, 0);
  heatmapCtx.filter = 'none';

  // Blend heatmap onto main canvas
  ctx.globalAlpha = 0.6;
  ctx.drawImage(heatmapCanvas, 0, 0);
  ctx.globalAlpha = 1.0;

  // Draw debug landmarks if enabled
  if (showDebug) {
    drawDebugLandmarks(ctx, currentLandmarks, width, height);
  }

  return tensionScore;
}

function drawDebugLandmarks(
  ctx: CanvasRenderingContext2D,
  landmarks: Landmark[],
  width: number,
  height: number
): void {
  ctx.fillStyle = 'lime';
  ctx.font = '8px monospace';

  // Draw subset of landmarks with indices for debugging
  const debugIndices = [
    0, 13, 14, 33, 61, 70, 152, 263, 291, 300, // Key points
    ...Array.from({ length: 10 }, (_, i) => i * 47) // Every 47th landmark
  ];

  for (const i of debugIndices) {
    if (i >= landmarks.length) continue;
    
    const landmark = landmarks[i];
    const x = landmark.x * width;
    const y = landmark.y * height;

    // Draw point
    ctx.beginPath();
    ctx.arc(x, y, 2, 0, 2 * Math.PI);
    ctx.fill();

    // Draw index number
    ctx.fillText(i.toString(), x + 3, y - 3);
  }
}

