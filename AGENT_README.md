# Cerebras AI Agent Feature

## 🤖 AI-Powered Emotional Interview System

The app now includes an intelligent interview agent powered by **Cerebras Cloud SDK** that conducts psychological assessments with real-time emotion detection.

## Features

### 🎯 Intelligent Questioning
- **5 Initial Questions**: AI generates thought-provoking questions tailored to emotional assessment
- **Dynamic Follow-ups**: Detects emotional inconsistencies and asks deeper questions
- **Adaptive Flow**: Adjusts questioning based on your responses and detected emotions

### 📊 Real-Time Analysis
- **Emotion Detection**: Captures your facial expression during each response
- **Tension Monitoring**: Records facial tension levels
- **Visual Context**: Saves screenshot of your face state
- **Audio Transcription**: Optional voice input support

### 🧠 Comprehensive Reports
Generated reports include:
- **Emotional Authenticity**: Do your words match your facial expressions?
- **Stress Indicators**: High tension periods and emotion fluctuations
- **Emotional Stability**: How consistent were your emotions?
- **Key Findings**: Psychological insights from the AI
- **Recommendations**: Actionable suggestions based on analysis

## Setup

### 1. Get Cerebras API Key
Visit [cloud.cerebras.ai](https://cloud.cerebras.ai) and sign up for an API key.

### 2. Start Interview
1. Click "🤖 Start AI Interview" button
2. Enter your Cerebras API key when prompted
3. Answer questions naturally
4. Use voice input or type responses

## How It Works

### Interview Flow
```
1. AI generates 5 initial probing questions
2. For each question:
   - You respond (text or voice)
   - System captures:
     * Your response text
     * Detected emotion from face
     * Facial tension level
     * Screenshot of your face
   - AI analyzes response + emotion
   - May ask follow-up if inconsistency detected
3. After all questions:
   - AI analyzes entire session
   - Generates comprehensive JSON report
```

### Example Question Sequence
```
Q1: "How are you feeling right now?"
→ User says "I'm fine" but shows 😢 Sad
→ AI detects inconsistency
→ Follow-up: "You mentioned feeling fine, but I noticed some tension. Is there something bothering you?"
```

### Smart Detection
The AI looks for:
- **Emotional Mismatches**: Saying "happy" while looking sad
- **Evasive Answers**: Short, vague responses to deep questions
- **High Stress**: Elevated facial tension scores
- **Emotion Shifts**: Rapid changes in detected emotions

## Using Voice Input

If supported by your browser:
1. Click 🎤 Voice button
2. Speak your response
3. Text will auto-populate
4. Click Submit

## Report Structure

```json
{
  "sessionId": "session_1234567890",
  "startTime": 1234567890,
  "endTime": 1234567900,
  "responses": [
    {
      "questionId": "q1",
      "question": "How are you feeling?",
      "response": "I'm okay",
      "emotion": "Sad",
      "emotionConfidence": 0.85,
      "facialTension": 45.2,
      "imageData": "data:image/jpeg;base64,..."
    }
  ],
  "emotionalJourney": [
    {"timestamp": 1234567890, "emotion": "Neutral", "confidence": 0.9},
    {"timestamp": 1234567892, "emotion": "Sad", "confidence": 0.85}
  ],
  "insights": {
    "dominantEmotion": "Sad",
    "emotionalStability": 0.65,
    "stressLevel": 0.45,
    "authenticity": 0.70,
    "keyFindings": [
      "User displayed sad emotions while verbalizing positive sentiments",
      "Elevated stress during questions about personal topics"
    ],
    "recommendations": [
      "Consider exploring feelings of sadness further",
      "Practice emotional awareness exercises"
    ]
  },
  "summary": "Session showed disconnect between verbal and emotional expression..."
}
```

## Models Used

- **Question Generation**: `llama3.1-8b`
- **Follow-up Analysis**: `llama3.1-8b`
- **Report Generation**: `llama3.1-8b`

All powered by **Cerebras Cloud** for ultra-fast inference!

## Privacy

- All data processed client-side except AI inference
- Images stored in report JSON (can be excluded)
- No data sent to servers except Cerebras API
- Reports saved locally as JSON files

## Tips for Best Results

1. **Be Natural**: Answer honestly and naturally
2. **Good Lighting**: Ensures accurate emotion detection
3. **Face Camera**: Keep face visible during responses
4. **Take Your Time**: No rush, think before answering
5. **Be Honest**: Authentic responses yield better insights

## Troubleshooting

**"Speech recognition not supported"**
- Use Chrome, Edge, or Safari
- Or type responses manually

**"No face detected"**
- Ensure camera is active
- Check lighting and positioning

**"Failed to initialize Cerebras"**
- Verify API key is correct
- Check internet connection
- Ensure API key has credits

## Example Use Cases

1. **Emotional Check-ins**: Daily mood assessments
2. **Stress Monitoring**: Track stress over time
3. **Therapy Sessions**: Supplement professional therapy
4. **Self-Awareness**: Understand your emotional patterns
5. **Research**: Collect emotion data with consent

---

**Ready to explore your emotions with AI! 🤖💭🧠**
