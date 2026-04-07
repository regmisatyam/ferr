import { InteractionReport, EmotionLabel } from '../types';

interface MetricsData {
  distressLevel: { before: number; after: number };
  stressLevel: { before: number; after: number };
  emotionalStability: { before: number; after: number };
  authenticity: { before: number; after: number };
}

// Calculate distress level from emotion and tension
function calculateDistressLevel(emotion: EmotionLabel, tension: number, confidence: number): number {
  const emotionDistressMap: Record<EmotionLabel, number> = {
    'Neutral': 0.2,
    'Happy': 0.1,
    'Sad': 0.7,
    'Angry': 0.8,
    'Fear': 0.9,
    'Surprise': 0.4,
    'Disgust': 0.6
  };
  
  const baseDistress = emotionDistressMap[emotion] || 0.5;
  const tensionFactor = Math.min(tension / 100, 1); // Normalize tension to 0-1
  
  // Weighted combination: 60% emotion, 40% tension
  return Math.min((baseDistress * 0.6 + tensionFactor * 0.4) * confidence, 1);
}

// Extract before/after metrics from report
function extractMetrics(report: InteractionReport): MetricsData {
  const responses = report.responses;
  const journey = report.emotionalJourney;
  
  // Calculate before metrics (first 20% of responses)
  const beforeCount = Math.max(1, Math.floor(responses.length * 0.2));
  const beforeResponses = responses.slice(0, beforeCount);
  const beforeJourney = journey.slice(0, Math.max(1, Math.floor(journey.length * 0.2)));
  
  // Calculate after metrics (last 20% of responses)
  const afterCount = Math.max(1, Math.floor(responses.length * 0.2));
  const afterResponses = responses.slice(-afterCount);
  const afterJourney = journey.slice(-Math.max(1, Math.floor(journey.length * 0.2)));
  
  // Before distress level
  const beforeDistress = beforeResponses.reduce((sum, r) => 
    sum + calculateDistressLevel(r.emotion, r.facialTension, r.emotionConfidence), 0
  ) / beforeResponses.length;
  
  // After distress level
  const afterDistress = afterResponses.reduce((sum, r) => 
    sum + calculateDistressLevel(r.emotion, r.facialTension, r.emotionConfidence), 0
  ) / afterResponses.length;
  
  return {
    distressLevel: {
      before: beforeDistress,
      after: afterDistress
    },
    stressLevel: {
      before: beforeResponses.reduce((sum, r) => sum + r.facialTension, 0) / beforeResponses.length / 100,
      after: afterResponses.reduce((sum, r) => sum + r.facialTension, 0) / afterResponses.length / 100
    },
    emotionalStability: {
      before: calculateStability(beforeJourney),
      after: calculateStability(afterJourney)
    },
    authenticity: {
      before: report.insights.authenticity, // Use overall metric
      after: report.insights.authenticity
    }
  };
}

// Calculate emotional stability from journey
function calculateStability(journey: { emotion: EmotionLabel; confidence: number }[]): number {
  if (journey.length < 2) return 1;
  
  let changes = 0;
  for (let i = 1; i < journey.length; i++) {
    if (journey[i].emotion !== journey[i - 1].emotion) {
      changes++;
    }
  }
  
  return Math.max(0, 1 - (changes / journey.length));
}

// Generate line chart on canvas
export function generateMetricsChart(report: InteractionReport): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = 1200;
  canvas.height = 800;
  
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Could not get canvas context');
  
  const metrics = extractMetrics(report);
  
  // Background
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  // Margins and dimensions
  const margin = { top: 80, right: 200, bottom: 80, left: 80 };
  const chartWidth = canvas.width - margin.left - margin.right;
  const chartHeight = canvas.height - margin.top - margin.bottom;
  
  // Title
  ctx.fillStyle = '#000000';
  ctx.font = 'bold 28px Arial';
  ctx.textAlign = 'center';
  ctx.fillText('Before & After Interview Metrics Comparison', canvas.width / 2, 40);
  
  // Chart area background
  ctx.fillStyle = '#f9f9f9';
  ctx.fillRect(margin.left, margin.top, chartWidth, chartHeight);
  
  // Grid lines
  ctx.strokeStyle = '#e0e0e0';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 10; i++) {
    const y = margin.top + (chartHeight * i / 10);
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + chartWidth, y);
    ctx.stroke();
  }
  
  // Y-axis labels
  ctx.fillStyle = '#666666';
  ctx.font = '14px Arial';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 10; i++) {
    const y = margin.top + (chartHeight * i / 10);
    const value = ((10 - i) * 10).toString() + '%';
    ctx.fillText(value, margin.left - 10, y + 5);
  }
  
  // X-axis labels
  ctx.textAlign = 'center';
  const xPositions = [
    margin.left + chartWidth * 0.25,
    margin.left + chartWidth * 0.75
  ];
  const labels = ['Before', 'After'];
  
  ctx.font = 'bold 18px Arial';
  ctx.fillStyle = '#000000';
  labels.forEach((label, i) => {
    ctx.fillText(label, xPositions[i], margin.top + chartHeight + 40);
  });
  
  // Draw lines and data points
  const dataPoints: Array<{ metric: string; color: string; data: { before: number; after: number } }> = [
    { metric: 'Stress Level', color: '#ff6b6b', data: metrics.stressLevel },
    { metric: 'Emotional Instability', color: '#4ecdc4', data: { 
      before: 1 - metrics.emotionalStability.before,
      after: 1 - metrics.emotionalStability.after 
    }},
    { metric: 'Inauthenticity', color: '#95a5a6', data: {
      before: 1 - metrics.authenticity.before,
      after: 1 - metrics.authenticity.after
    }},
  ];
  
  // Draw other metrics first
  dataPoints.forEach(({ metric, color, data }) => {
    const beforeY = margin.top + chartHeight * (1 - data.before);
    const afterY = margin.top + chartHeight * (1 - data.after);
    
    // Draw line
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(xPositions[0], beforeY);
    ctx.lineTo(xPositions[1], afterY);
    ctx.stroke();
    
    // Draw points
    [xPositions[0], xPositions[1]].forEach((x, i) => {
      const y = i === 0 ? beforeY : afterY;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fill();
      
      // Draw value labels
      ctx.fillStyle = '#000000';
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      const value = ((i === 0 ? data.before : data.after) * 100).toFixed(0) + '%';
      ctx.fillText(value, x, y - 15);
    });
  });
  
  // Draw distress level (yellow, thicker, separate)
  const distressBefore = margin.top + chartHeight * (1 - metrics.distressLevel.before);
  const distressAfter = margin.top + chartHeight * (1 - metrics.distressLevel.after);
  
  ctx.strokeStyle = '#FFD700';
  ctx.lineWidth = 5;
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(xPositions[0], distressBefore);
  ctx.lineTo(xPositions[1], distressAfter);
  ctx.stroke();
  
  // Draw distress points (larger)
  [xPositions[0], xPositions[1]].forEach((x, i) => {
    const y = i === 0 ? distressBefore : distressAfter;
    
    // Outer glow
    ctx.fillStyle = 'rgba(255, 215, 0, 0.3)';
    ctx.beginPath();
    ctx.arc(x, y, 12, 0, Math.PI * 2);
    ctx.fill();
    
    // Inner circle
    ctx.fillStyle = '#FFD700';
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.fill();
    
    // Black border
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.stroke();
    
    // Draw value labels
    ctx.fillStyle = '#000000';
    ctx.font = 'bold 14px Arial';
    ctx.textAlign = 'center';
    const value = ((i === 0 ? metrics.distressLevel.before : metrics.distressLevel.after) * 100).toFixed(0) + '%';
    ctx.fillText(value, x, y - 20);
  });
  
  // Legend
  const legendX = margin.left + chartWidth + 20;
  const legendY = margin.top;
  
  ctx.font = 'bold 16px Arial';
  ctx.fillStyle = '#000000';
  ctx.textAlign = 'left';
  ctx.fillText('Metrics', legendX, legendY);
  
  // Distress level legend (first, prominent)
  let currentY = legendY + 30;
  ctx.fillStyle = '#FFD700';
  ctx.fillRect(legendX, currentY - 10, 30, 4);
  ctx.fillStyle = '#000000';
  ctx.font = 'bold 14px Arial';
  ctx.fillText('Distress Level', legendX + 40, currentY);
  currentY += 30;
  
  // Other metrics legend
  dataPoints.forEach(({ metric, color }) => {
    ctx.fillStyle = color;
    ctx.fillRect(legendX, currentY - 8, 20, 3);
    ctx.fillStyle = '#000000';
    ctx.font = '14px Arial';
    ctx.fillText(metric, legendX + 30, currentY);
    currentY += 25;
  });
  
  // Add note about distress
  ctx.font = 'italic 12px Arial';
  ctx.fillStyle = '#666666';
  ctx.textAlign = 'left';
  const noteY = margin.top + chartHeight + 65;
  ctx.fillText('Note: Distress level combines emotional state and facial tension', margin.left, noteY);
  
  // Axes
  ctx.strokeStyle = '#000000';
  ctx.lineWidth = 2;
  
  // Y-axis
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + chartHeight);
  ctx.stroke();
  
  // X-axis
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top + chartHeight);
  ctx.lineTo(margin.left + chartWidth, margin.top + chartHeight);
  ctx.stroke();
  
  // Y-axis label
  ctx.save();
  ctx.translate(20, canvas.height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.font = 'bold 16px Arial';
  ctx.fillStyle = '#000000';
  ctx.textAlign = 'center';
  ctx.fillText('Metric Level (%)', 0, 0);
  ctx.restore();
  
  return canvas;
}

// Download chart as PNG
export function downloadMetricsChart(report: InteractionReport): void {
  const canvas = generateMetricsChart(report);
  
  canvas.toBlob((blob) => {
    if (!blob) {
      console.error('Failed to generate chart blob');
      return;
    }
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `metrics-comparison-${report.sessionId}.png`;
    a.click();
    URL.revokeObjectURL(url);
  }, 'image/png');
}
