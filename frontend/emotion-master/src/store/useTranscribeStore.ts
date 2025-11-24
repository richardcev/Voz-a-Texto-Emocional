import { create } from 'zustand';
import type { KaraokeEmotionResponse } from '@/types/karaokeEmotion';

type State = {
  audioFile: File | null;
  audioUrl: string | null;
  data: KaraokeEmotionResponse | null;

  currentTime: number;
  audioEl: HTMLAudioElement | null;
};

type Actions = {
  setAudioFile: (file: File | null) => void; // <-- NUEVO
  setAudioUrl: (url: string | null) => void;
  setData: (d: KaraokeEmotionResponse | null) => void;

  setCurrentTime: (t: number) => void;
  setAudioEl: (el: HTMLAudioElement | null) => void;

  seek: (t: number) => void;
  play: () => void;
  pause: () => void;
};

export const useTranscribeStore = create<State & Actions>((set, get) => ({
  audioFile: null,
  audioUrl: null,
  data: null,

  currentTime: 0,
  audioEl: null,

  // Crea un blob URL y revoca el anterior si existía
  setAudioFile: file => {
    const prevUrl = get().audioUrl;
    if (prevUrl && prevUrl.startsWith('blob:')) {
      try {
        URL.revokeObjectURL(prevUrl);
      } catch { /* empty */ }
    }

    if (file) {
      const url = URL.createObjectURL(file);
      set({ audioFile: file, audioUrl: url, data: null, currentTime: 0 });
    } else {
      set({ audioFile: null, audioUrl: null, data: null, currentTime: 0 });
    }
  },

  setAudioUrl: url => set({ audioUrl: url }),
  setData: d => set({ data: d }),

  setCurrentTime: t => set({ currentTime: t }),
  setAudioEl: el => set({ audioEl: el }),

  seek: (t: number) => {
    const el = get().audioEl;
    const tt = Math.max(0, t);
    if (el) el.currentTime = tt;
    set({ currentTime: tt });
  },

  play: () => {
    const el = get().audioEl;
    if (el && el.paused) el.play().catch(() => {});
  },
  pause: () => {
    const el = get().audioEl;
    if (el && !el.paused) el.pause();
  },
}));
