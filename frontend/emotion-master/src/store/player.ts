import { create } from 'zustand';
import type { Segment, TranscribeResponse } from '@/types';

type PlayerState = {
  audioUrl: string | null;
  isPlaying: boolean;
  currentTime: number;
  duration: number;

  segments: Segment[];
  transcription: string;
  globalEmotions: { label: string; score: number }[];

  setAudioUrl: (url: string | null) => void;
  setPlaying: (v: boolean) => void;
  setTime: (t: number) => void;
  setDuration: (d: number) => void;
  setFromResponse: (r: TranscribeResponse) => void;
  clear: () => void;
};

export const usePlayerStore = create<PlayerState>(set => ({
  audioUrl: null,
  isPlaying: false,
  currentTime: 0,
  duration: 0,

  segments: [],
  transcription: '',
  globalEmotions: [],

  setAudioUrl: url => set({ audioUrl: url }),
  setPlaying: v => set({ isPlaying: v }),
  setTime: t => set({ currentTime: t }),
  setDuration: d => set({ duration: d }),
  setFromResponse: r =>
    set({
      segments: r.segments ?? [],
      transcription: r.transcription ?? '',
      globalEmotions: r.global_emotions ?? [],
    }),
  clear: () =>
    set({
      audioUrl: null,
      isPlaying: false,
      currentTime: 0,
      duration: 0,
      segments: [],
      transcription: '',
      globalEmotions: [],
    }),
}));
