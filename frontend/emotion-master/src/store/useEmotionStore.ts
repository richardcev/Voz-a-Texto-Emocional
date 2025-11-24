import { create } from 'zustand';

export type Emotion = 'happy' | 'sad' | 'angry' | 'calm';

interface EmotionState {
  current: Emotion;
  note: string;
  setEmotion: (emotion: Emotion) => void;
  setNote: (note: string) => void;
  reset: () => void;
}

export const useEmotionStore = create<EmotionState>(set => ({
  current: 'happy',
  note: '',
  setEmotion: emotion => set({ current: emotion }),
  setNote: note => set({ note }),
  reset: () => set({ current: 'happy', note: '' }),
}));
