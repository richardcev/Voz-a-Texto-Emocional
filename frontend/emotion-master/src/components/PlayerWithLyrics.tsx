import { useEffect, useMemo, useRef, useState } from 'react';
import { useTranscribeStore } from '@/store/useTranscribeStore';
import type { WordTS } from '@/types/karaokeEmotion';

function within(t: number, w: WordTS) {
  return t >= w.start && t < w.end;
}

function findActiveIndex(words: WordTS[], t: number) {
  let lo = 0,
    hi = words.length - 1,
    ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const w = words[mid];
    if (t < w.start) hi = mid - 1;
    else if (t >= w.end) lo = mid + 1;
    else {
      ans = mid;
      break;
    }
  }
  return ans;
}

export function PlayerWithLyrics() {
  const audioUrl = useTranscribeStore(s => s.audioUrl);
  const data = useTranscribeStore(s => s.data);
  const setCurrentTime = useTranscribeStore(s => s.setCurrentTime);
  const currentTime = useTranscribeStore(s => s.currentTime);
  const setAudioEl = useTranscribeStore(s => s.setAudioEl);

  const audioRef = useRef<HTMLAudioElement>(null);
  const lyricsRef = useRef<HTMLDivElement>(null);
  const [rafOn, setRafOn] = useState(false);

  // flatten de palabras (orden natural)
  const words = useMemo<WordTS[]>(() => {
    const out: WordTS[] = [];
    (data?.segments ?? []).forEach(s => s.words && out.push(...s.words));
    return out;
  }, [data]);

  // conecta <audio> al store (una sola fuente de la verdad)
  useEffect(() => {
    if (audioRef.current) setAudioEl(audioRef.current);
    return () => setAudioEl(null);
  }, [setAudioEl]);

  // timeupdate (respaldo) + rAF (suavidad)
  useEffect(() => {
    const a = audioRef.current;
    if (!a) return;

    let rafId = 0;
    const onTime = () => setCurrentTime(a.currentTime);

    const tick = () => {
      setCurrentTime(a.currentTime);
      rafId = requestAnimationFrame(tick);
    };

    a.addEventListener('timeupdate', onTime);

    const onPlay = () => {
      if (!rafOn) {
        setRafOn(true);
        rafId = requestAnimationFrame(tick);
      }
    };
    const onPause = () => {
      setRafOn(false);
      cancelAnimationFrame(rafId);
    };
    a.addEventListener('play', onPlay);
    a.addEventListener('pause', onPause);
    a.addEventListener('ended', onPause);

    return () => {
      a.removeEventListener('timeupdate', onTime);
      a.removeEventListener('play', onPlay);
      a.removeEventListener('pause', onPause);
      a.removeEventListener('ended', onPause);
      cancelAnimationFrame(rafId);
    };
  }, [setCurrentTime, rafOn]);

  // índice activo y autoscroll suave dentro del párrafo
  const activeIdx = useMemo(
    () => findActiveIndex(words, currentTime),
    [words, currentTime]
  );

  useEffect(() => {
    if (activeIdx < 0) return;
    const container = lyricsRef.current;
    if (!container) return;

    const span = container.querySelector<HTMLSpanElement>(
      `[data-w="${activeIdx}"]`
    );
    if (!span) return;

    const cRect = container.getBoundingClientRect();
    const sRect = span.getBoundingClientRect();
    const isVisible = sRect.top >= cRect.top && sRect.bottom <= cRect.bottom;

    if (!isVisible) {
      span.scrollIntoView({
        behavior: 'smooth',
        block: 'center',
        inline: 'nearest',
      });
    }
  }, [activeIdx]);

  if (!audioUrl) return null;

  return (
    <div className="space-y-3">
      <audio ref={audioRef} src={audioUrl} controls className="w-full" />
      <div
        ref={lyricsRef}
        className="rounded-lg border p-4 leading-8 max-h-60 overflow-y-auto"
      >
        {words.map((w, idx) => {
          const isActive = idx === activeIdx || within(currentTime, w);
          return (
            <span
              key={idx}
              data-w={idx}
              className={
                isActive ? 'bg-foreground text-background rounded px-1' : ''
              }
            >
              {w.text + ' '}
            </span>
          );
        })}
      </div>
    </div>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export function useSeekFromSegments() {
  const seek = useTranscribeStore(s => s.seek);
  return (t: number) => seek(t);
}
