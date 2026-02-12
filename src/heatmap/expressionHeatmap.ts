import { FaceExpressions } from '../types';

// Color map for different emotions
const EMOTION_COLORS: Record<string, [number, number, number]> = {
  neutral: [200, 200, 200],  // Gray
  happy: [255, 215, 0],      // Gold
  sad: [0, 100, 255],        // Blue
  angry: [255, 0, 0],        // Red
  fearful: [138, 43, 226],   // Purple
  disgusted: [0, 255, 0],    // Green
  surprised: [255, 165, 0]   // Orange
};

export function renderExpressionHeatmap(
  canvas: HTMLCanvasElement,
  videoElement: HTMLVideoElement,
  expressions: FaceExpressions | null
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx || !expressions) return;

  const width = canvas.width;
  const height = canvas.height;

  // Create gradient overlay based on expression intensities
  const overlayCanvas = document.createElement('canvas');
  overlayCanvas.width = width;
  overlayCanvas.height = height;
  const overlayCtx = overlayCanvas.getContext('2d');
  if (!overlayCtx) return;

  // Blend colors based on expression intensities
  const emotions = [
    { name: 'neutral', value: expressions.neutral },
    { name: 'happy', value: expressions.happy },
    { name: 'sad', value: expressions.sad },
    { name: 'angry', value: expressions.angry },
    { name: 'fearful', value: expressions.fearful },
    { name: 'disgusted', value: expressions.disgusted },
    { name: 'surprised', value: expressions.surprised }
  ];

  // Sort by intensity
  emotions.sort((a, b) => b.value - a.value);

  // Create radial gradient from center
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.max(width, height) / 2;

  // Blend top 3 emotions
  for (let i = 0; i < Math.min(3, emotions.length); i++) {
    const emotion = emotions[i];
    if (emotion.value < 0.1) continue; // Skip very weak emotions

    const color = EMOTION_COLORS[emotion.name];
    const gradient = overlayCtx.createRadialGradient(
      centerX, centerY, 0,
      centerX, centerY, radius
    );

    const alpha = emotion.value * 0.3; // Scale alpha
    gradient.addColorStop(0, `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${alpha})`);
    gradient.addColorStop(1, `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0)`);

    overlayCtx.fillStyle = gradient;
    overlayCtx.fillRect(0, 0, width, height);
  }

  // Apply blur for smoother effect
  overlayCtx.filter = 'blur(40px)';
  overlayCtx.drawImage(overlayCanvas, 0, 0);
  overlayCtx.filter = 'none';

  // Blend onto main canvas
  ctx.globalAlpha = 0.5;
  ctx.globalCompositeOperation = 'screen';
  ctx.drawImage(overlayCanvas, 0, 0);
  ctx.globalAlpha = 1.0;
  ctx.globalCompositeOperation = 'source-over';
}

export function renderExpressionBars(
  canvas: HTMLCanvasElement,
  expressions: FaceExpressions
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const width = canvas.width;
  const height = canvas.height;
  const barHeight = 20;
  const padding = 10;
  const startY = height - 200;

  const emotions = [
    { name: 'Neutral', key: 'neutral', value: expressions.neutral },
    { name: 'Happy', key: 'happy', value: expressions.happy },
    { name: 'Sad', key: 'sad', value: expressions.sad },
    { name: 'Angry', key: 'angry', value: expressions.angry },
    { name: 'Fear', key: 'fearful', value: expressions.fearful },
    { name: 'Disgust', key: 'disgusted', value: expressions.disgusted },
    { name: 'Surprise', key: 'surprised', value: expressions.surprised }
  ];

  // Draw semi-transparent background
  ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
  ctx.fillRect(padding, startY - padding, 250, emotions.length * (barHeight + 5) + padding * 2);

  emotions.forEach((emotion, index) => {
    const y = startY + index * (barHeight + 5);
    const color = EMOTION_COLORS[emotion.key];
    
    // Label
    ctx.fillStyle = 'white';
    ctx.font = '12px monospace';
    ctx.fillText(emotion.name, padding + 5, y + 15);

    // Bar background
    ctx.fillStyle = 'rgba(100, 100, 100, 0.5)';
    ctx.fillRect(padding + 80, y, 150, barHeight);

    // Bar fill
    const barWidth = emotion.value * 150;
    ctx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    ctx.fillRect(padding + 80, y, barWidth, barHeight);

    // Percentage
    ctx.fillStyle = 'white';
    ctx.fillText(`${(emotion.value * 100).toFixed(0)}%`, padding + 235, y + 15);
  });
}

