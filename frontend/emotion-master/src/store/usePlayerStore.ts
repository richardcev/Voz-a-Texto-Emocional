import { create } from 'zustand';
import type { KaraokeResponse, KaraokeSegment } from '@/types';

type PlayerState = {
  audioUrl: string | null;
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  transcription: string;
  segments: KaraokeSegment[];
  backend: 'faster-whisper' | 'whisper' | null;

  setAudioUrl: (url: string | null) => void;
  setIsPlaying: (v: boolean) => void;
  setCurrentTime: (t: number) => void;
  setDuration: (d: number) => void;

  loadKaraoke: (payload: KaraokeResponse) => void;
  reset: () => void;
};

export const usePlayerStore = create<PlayerState>(set => ({
  audioUrl: null,
  isPlaying: false,
  currentTime: 0,
  duration: 0,
  transcription: '',
  segments: [],
  backend: null,

  setAudioUrl: url => set({ audioUrl: url }),
  setIsPlaying: v => set({ isPlaying: v }),
  setCurrentTime: t => set({ currentTime: t }),
  setDuration: d => set({ duration: d }),

  loadKaraoke: payload =>
    set({
      transcription: payload.transcription,
      segments: payload.segments,
      duration: payload.duration ?? 0,
      backend: payload.backend,
    }),

  reset: () =>
    set({
      audioUrl: null,
      isPlaying: false,
      currentTime: 0,
      duration: 0,
      transcription: '',
      segments: [],
      backend: null,
    }),
}));
