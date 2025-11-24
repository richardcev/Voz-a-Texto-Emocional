export type EmotionScore = { label: string; score: number };

export type Segment = {
  start: number;
  end: number;
  text: string;
  top_emotion: EmotionScore;
  emotions: EmotionScore[];
};

export type TranscribeResponse = {
  transcription: string;
  global_emotions: EmotionScore[];
  top_global_emotions: EmotionScore[];
  segments: Segment[];
};

export type KaraokeWord = {
  text: string;
  start: number;
  end: number;
};

export type KaraokeSegment = {
  start: number;
  end: number;
  text: string;
  words: KaraokeWord[];
};

export type KaraokeResponse = {
  backend: 'faster-whisper' | 'whisper';
  transcription: string;
  segments: KaraokeSegment[];
  duration: number | null;
  language: string;
};


