// Speech recognition for capturing user responses

let recognition: any = null;
let isListening = false;

export function initSpeechRecognition(): boolean {
  // Check for browser support
  const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
  
  if (!SpeechRecognition) {
    console.warn('Speech recognition not supported in this browser');
    return false;
  }

  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.lang = 'en-US';

  return true;
}

export function startListening(): Promise<string> {
  return new Promise((resolve, reject) => {
    if (!recognition) {
      reject(new Error('Speech recognition not initialized'));
      return;
    }

    if (isListening) {
      reject(new Error('Already listening'));
      return;
    }

    isListening = true;
    let transcript = '';

    recognition.onresult = (event: any) => {
      transcript = event.results[0][0].transcript;
      console.log('Heard:', transcript);
    };

    recognition.onerror = (event: any) => {
      isListening = false;
      reject(new Error(`Speech recognition error: ${event.error}`));
    };

    recognition.onend = () => {
      isListening = false;
      resolve(transcript);
    };

    try {
      recognition.start();
    } catch (error) {
      isListening = false;
      reject(error);
    }
  });
}

export function stopListening(): void {
  if (recognition && isListening) {
    recognition.stop();
    isListening = false;
  }
}

export function isCurrentlyListening(): boolean {
  return isListening;
}

export function isSpeechRecognitionSupported(): boolean {
  return 'SpeechRecognition' in window || 'webkitSpeechRecognition' in window;
}
