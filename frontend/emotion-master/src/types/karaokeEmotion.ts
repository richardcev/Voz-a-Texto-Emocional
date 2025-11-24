export type WordTS = { text: string; start: number; end: number };

export type EmotionScore = { label: string; score: number };

export type SegmentKE = {
  start: number;
  end: number;
  text: string;
  words: WordTS[];
  top_emotion: EmotionScore | null;
  emotions: EmotionScore[];
};

export type KaraokeEmotionResponse = {
  backend: 'faster-whisper' | 'whisper';
  transcription: string;
  segments: SegmentKE[];
  global_emotions: EmotionScore[];
  top_global_emotions: EmotionScore[];
  duration: number | null;
  language: string;
};
