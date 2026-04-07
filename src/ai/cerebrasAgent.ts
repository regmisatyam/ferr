import Cerebras from '@cerebras/cerebras_cloud_sdk';
import { AgenticQuestion, UserResponse, InteractionReport, EmotionLabel } from '../types';

// Cerebras client
let cerebras: Cerebras | null = null;

export function initCerebrasAgent(apiKey: string): void {
  cerebras = new Cerebras({
    apiKey: apiKey,
  });
}

export function isAgentInitialized(): boolean {
  return cerebras !== null;
}

// Generate initial probing questions
export async function generateProbingQuestions(
  context?: string
): Promise<AgenticQuestion[]> {
  if (!cerebras) throw new Error('Cerebras agent not initialized');

  const systemPrompt = `You are an empathetic psychological interviewer. Generate 5 thought-provoking questions that:
1. Start easy and gradually get deeper
2. Explore emotional states and authenticity
3. Can reveal stress, deception, or emotional conflicts
4. Are conversational and non-threatening
5. Mix different question types (open, yes/no, scale)

Context: ${context || 'General emotional assessment session'}

Return ONLY a JSON array of questions with this structure:
[
  {
    "id": "q1",
    "question": "How would you describe your current mood?",
    "type": "open"
  },
  {
    "id": "q2", 
    "question": "On a scale of 1-10, how stressed do you feel right now?",
    "type": "scale"
  }
]`;

  try {
    const response = await cerebras.chat.completions.create({
      model: 'llama3.1-8b',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: 'Generate the questions.' }
      ],
      temperature: 0.8,
      max_tokens: 1000
    });

    // Handle possible 'unknown' type for response.choices
    let content = '[]';
    if (
      response &&
      typeof response === 'object' &&
      'choices' in response &&
      Array.isArray((response as any).choices) &&
      (response as any).choices.length > 0 &&
      (response as any).choices[0]?.message?.content
    ) {
      content = (response as any).choices[0].message.content;
    }
    const questions = JSON.parse(content);
    return questions;
  } catch (error) {
    console.error('Error generating questions:', error);
    // Fallback questions
    return [
      { id: 'q1', question: 'How are you feeling right now?', type: 'open' },
      { id: 'q2', question: 'Is there anything making you uncomfortable today?', type: 'yesno' },
      { id: 'q3', question: 'On a scale of 1-10, how would you rate your stress level?', type: 'scale' },
      { id: 'q4', question: 'What emotion would you say is most present for you?', type: 'open' },
      { id: 'q5', question: 'Are you being completely honest about your feelings?', type: 'yesno' }
    ];
  }
}

// Generate follow-up question based on previous response
export async function generateFollowUpQuestion(
  previousResponse: UserResponse,
  allResponses: UserResponse[]
): Promise<AgenticQuestion | null> {
  if (!cerebras) throw new Error('Cerebras agent not initialized');

  const emotionContext = allResponses.map(r => 
    `Q: ${r.question}\nA: ${r.response}\nDetected Emotion: ${r.emotion} (${(r.emotionConfidence * 100).toFixed(0)}%)`
  ).join('\n\n');

  const systemPrompt = `You are a perceptive interviewer analyzing responses. Based on the conversation history and detected emotions, generate ONE insightful follow-up question.

Conversation History:
${emotionContext}

Latest Response:
Q: ${previousResponse.question}
A: ${previousResponse.response}
Detected Emotion: ${previousResponse.emotion} (${(previousResponse.emotionConfidence * 100).toFixed(0)}%)

Look for:
- Emotional inconsistencies (saying "fine" but showing sad/angry emotions)
- Evasive answers
- High tension or stress indicators
- Opportunities to dig deeper

Generate a follow-up question that gently probes these areas.

Return ONLY a JSON object:
{
  "id": "followup_X",
  "question": "Your follow-up question here",
  "type": "open",
  "followUp": true
}`;

  try {
    const response = await cerebras.chat.completions.create({
      model: 'llama3.1-8b',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: 'Generate the follow-up question.' }
      ],
      temperature: 0.7,
      max_tokens: 200
    });

    // Fix: response.choices is of type unknown, so we need to parse it safely
    let content = '';
    try {
      // Try to safely access the LLM response content
      // @ts-ignore because the API type is unknown
      content = (response as any)?.choices?.[0]?.message?.content || 'null';
      const question = JSON.parse(content);
      return question;
    } catch (parseError) {
      console.error('Error parsing follow-up question JSON:', parseError, '\nRaw content:', content);
      return null;
    }
  } catch (error) {
    console.error('Error generating follow-up:', error);
    return null;
  }
}

// Analyze responses and generate comprehensive report
export async function analyzeResponses(
  sessionId: string,
  responses: UserResponse[],
  emotionalJourney: { timestamp: number; emotion: EmotionLabel; confidence: number }[]
): Promise<InteractionReport> {
  if (!cerebras) throw new Error('Cerebras agent not initialized');

  const conversationSummary = responses.map((r, i) => 
    `${i + 1}. Q: ${r.question}\n   A: "${r.response}"\n   Emotion: ${r.emotion} (${(r.emotionConfidence * 100).toFixed(0)}%) | Tension: ${r.facialTension.toFixed(1)}`
  ).join('\n\n');

  const emotionSequence = emotionalJourney.map(e => 
    `${e.emotion}(${(e.confidence * 100).toFixed(0)}%)`
  ).join(' → ');

  const systemPrompt = `You are an expert psychologist analyzing an interview session with real-time emotion detection.

CONVERSATION TRANSCRIPT:
${conversationSummary}

EMOTIONAL JOURNEY:
${emotionSequence}

Analyze:
1. Emotional authenticity (do words match detected emotions?)
2. Stress indicators (high tension, emotion shifts)
3. Emotional stability (how much emotions fluctuated)
4. Key psychological insights
5. Potential areas of concern
6. Recommendations

Return a JSON object with this EXACT structure:
{
  "dominantEmotion": "Happy",
  "emotionalStability": 0.75,
  "stressLevel": 0.35,
  "authenticity": 0.80,
  "keyFindings": [
    "Finding 1",
    "Finding 2",
    "Finding 3"
  ],
  "recommendations": [
    "Recommendation 1",
    "Recommendation 2"
  ],
  "summary": "A brief 2-3 sentence summary of the entire session"
}

Values should be 0-1 floats. dominantEmotion should be one of: Neutral, Happy, Angry, Sad, Surprise, Fear, Disgust`;

  try {
    const response = await cerebras.chat.completions.create({
      model: 'llama3.1-8b',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: 'Analyze the session and return the JSON report.' }
      ],
      temperature: 0.6,
      max_tokens: 1500
    });

    const content = (response as any).choices[0]?.message?.content || '{}';
    const insights = JSON.parse(content);

    const report: InteractionReport = {
      sessionId,
      startTime: responses[0]?.timestamp || Date.now(),
      endTime: Date.now(),
      responses,
      emotionalJourney,
      insights,
      summary: insights.summary || 'Analysis complete.'
    };

    return report;
  } catch (error) {
    console.error('Error analyzing responses:', error);
    
    // Fallback analysis
    const emotionCounts: Record<string, number> = {};
    emotionalJourney.forEach(e => {
      emotionCounts[e.emotion] = (emotionCounts[e.emotion] || 0) + 1;
    });
    const dominantEmotion = Object.entries(emotionCounts)
      .sort(([, a], [, b]) => b - a)[0]?.[0] as EmotionLabel || 'Neutral';

    return {
      sessionId,
      startTime: responses[0]?.timestamp || Date.now(),
      endTime: Date.now(),
      responses,
      emotionalJourney,
      insights: {
        dominantEmotion,
        emotionalStability: 0.5,
        stressLevel: 0.5,
        authenticity: 0.5,
        keyFindings: ['Analysis completed with limited data'],
        recommendations: ['Consider re-running with Cerebras API key']
      },
      summary: 'Session analysis completed.'
    };
  }
}

// Export report as JSON
export function exportReport(report: InteractionReport): string {
  return JSON.stringify(report, null, 2);
}

// Export report with image data
export function exportReportWithImages(report: InteractionReport): string {
  // Include image data in responses
  return JSON.stringify(report, (key, value) => {
    if (key === 'imageData' && value) {
      return `[Image Data: ${value.substring(0, 50)}...]`;
    }
    return value;
  }, 2);
}
