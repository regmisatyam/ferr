import { Landmark, TrainingSample, ClassifierState, RegisteredFace } from '../types';

const DB_NAME = 'FaceExpressionDB';
const DB_VERSION = 2;
const BASELINE_STORE = 'baseline';
const SAMPLES_STORE = 'samples';
const CLASSIFIER_STORE = 'classifier';
const FACES_STORE = 'faces';

let db: IDBDatabase | null = null;

export async function initDB(): Promise<void> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      db = request.result;
      resolve();
    };

    request.onupgradeneeded = (event) => {
      const database = (event.target as IDBOpenDBRequest).result;

      // Store for baseline landmarks
      if (!database.objectStoreNames.contains(BASELINE_STORE)) {
        database.createObjectStore(BASELINE_STORE);
      }

      // Store for training samples
      if (!database.objectStoreNames.contains(SAMPLES_STORE)) {
        const samplesStore = database.createObjectStore(SAMPLES_STORE, { 
          keyPath: 'timestamp' 
        });
        samplesStore.createIndex('label', 'label', { unique: false });
      }

      // Store for classifier state
      if (!database.objectStoreNames.contains(CLASSIFIER_STORE)) {
        database.createObjectStore(CLASSIFIER_STORE);
      }

      // Store for registered faces
      if (!database.objectStoreNames.contains(FACES_STORE)) {
        const facesStore = database.createObjectStore(FACES_STORE, { 
          keyPath: 'name' 
        });
        facesStore.createIndex('lastSeen', 'lastSeen', { unique: false });
      }
    };
  });
}

function getDB(): IDBDatabase {
  if (!db) throw new Error('Database not initialized');
  return db;
}

// Baseline operations
export async function saveBaseline(landmarks: Landmark[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const transaction = getDB().transaction([BASELINE_STORE], 'readwrite');
    const store = transaction.objectStore(BASELINE_STORE);
    const request = store.put(landmarks, 'baseline');

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve();
  });
}

export async function loadBaseline(): Promise<Landmark[] | null> {
  return new Promise((resolve, reject) => {
    const transaction = getDB().transaction([BASELINE_STORE], 'readonly');
    const store = transaction.objectStore(BASELINE_STORE);
    const request = store.get('baseline');

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result || null);
  });
}

// Training samples operations
export async function saveSample(sample: TrainingSample): Promise<void> {
  return new Promise((resolve, reject) => {
    const transaction = getDB().transaction([SAMPLES_STORE], 'readwrite');
    const store = transaction.objectStore(SAMPLES_STORE);
    const request = store.add(sample);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve();
  });
}

export async function getAllSamples(): Promise<TrainingSample[]> {
  return new Promise((resolve, reject) => {
    const transaction = getDB().transaction([SAMPLES_STORE], 'readonly');
    const store = transaction.objectStore(SAMPLES_STORE);
    const request = store.getAll();

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result || []);
  });
}

export async function clearSamples(): Promise<void> {
  return new Promise((resolve, reject) => {
    const transaction = getDB().transaction([SAMPLES_STORE], 'readwrite');
    const store = transaction.objectStore(SAMPLES_STORE);
    const request = store.clear();

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve();
  });
}

// Classifier state operations
export async function saveClassifierState(state: ClassifierState): Promise<void> {
  return new Promise((resolve, reject) => {
    const transaction = getDB().transaction([CLASSIFIER_STORE], 'readwrite');
    const store = transaction.objectStore(CLASSIFIER_STORE);
    const request = store.put(state, 'classifier');

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve();
  });
}

export async function loadClassifierState(): Promise<ClassifierState | null> {
  return new Promise((resolve, reject) => {
    const transaction = getDB().transaction([CLASSIFIER_STORE], 'readonly');
    const store = transaction.objectStore(CLASSIFIER_STORE);
    const request = store.get('classifier');

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result || null);
  });
}

export async function clearClassifierState(): Promise<void> {
  return new Promise((resolve, reject) => {
    const transaction = getDB().transaction([CLASSIFIER_STORE], 'readwrite');
    const store = transaction.objectStore(CLASSIFIER_STORE);
    const request = store.clear();

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve();
  });
}

// Registered faces operations
export async function saveRegisteredFaces(faces: RegisteredFace[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const transaction = getDB().transaction([FACES_STORE], 'readwrite');
    const store = transaction.objectStore(FACES_STORE);
    
    // Clear existing and add all
    store.clear();
    
    for (const face of faces) {
      store.add(face);
    }

    transaction.onerror = () => reject(transaction.error);
    transaction.oncomplete = () => resolve();
  });
}

export async function loadRegisteredFaces(): Promise<RegisteredFace[]> {
  return new Promise((resolve, reject) => {
    const transaction = getDB().transaction([FACES_STORE], 'readonly');
    const store = transaction.objectStore(FACES_STORE);
    const request = store.getAll();

    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      const faces = request.result || [];
      // Convert descriptors back to Float32Array
      const converted = faces.map(face => ({
        ...face,
        descriptors: face.descriptors.map((d: any) => ({
          descriptor: new Float32Array(Object.values(d.descriptor)),
          timestamp: d.timestamp
        }))
      }));
      resolve(converted);
    };
  });
}

export async function clearRegisteredFaces(): Promise<void> {
  return new Promise((resolve, reject) => {
    const transaction = getDB().transaction([FACES_STORE], 'readwrite');
    const store = transaction.objectStore(FACES_STORE);
    const request = store.clear();

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve();
  });
}

// Export/Import functionality
export async function exportData(): Promise<string> {
  const baseline = await loadBaseline();
  const samples = await getAllSamples();
  const classifier = await loadClassifierState();
  const faces = await loadRegisteredFaces();

  const data = {
    baseline,
    samples,
    classifier,
    faces,
    exportDate: new Date().toISOString()
  };

  return JSON.stringify(data, null, 2);
}

export async function importData(jsonString: string): Promise<void> {
  try {
    const data = JSON.parse(jsonString);

    if (data.baseline) {
      await saveBaseline(data.baseline);
    }

    if (data.samples && Array.isArray(data.samples)) {
      await clearSamples();
      for (const sample of data.samples) {
        await saveSample(sample);
      }
    }

    if (data.classifier) {
      await saveClassifierState(data.classifier);
    }

    if (data.faces && Array.isArray(data.faces)) {
      await saveRegisteredFaces(data.faces);
    }
  } catch (error) {
    throw new Error('Invalid import data format');
  }
}

