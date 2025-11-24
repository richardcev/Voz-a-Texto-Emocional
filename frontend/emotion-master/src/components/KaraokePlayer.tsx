import { usePlayerStore } from '@/store/usePlayerStore';
import type { KaraokeSegment, KaraokeWord } from '@/types';
import { Pause, Play } from 'lucide-react';
import { useEffect, useMemo, useRef } from 'react';

function useRafTimeSync(onTick: () => void, enabled: boolean) {
  useEffect(() => {
    let raf = 0;
    const loop = () => {
      onTick();
      raf = requestAnimationFrame(loop);
    };
    if (enabled) raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [enabled, onTick]);
}

function classNames(...c: (string | false | undefined | null)[]) {
  return c.filter(Boolean).join(' ');
}

export default function KaraokePlayer() {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const {
    audioUrl,
    isPlaying,
    currentTime,
    duration,
    segments,
    setIsPlaying,
    setCurrentTime,
    setDuration,
  } = usePlayerStore();

  // Control fino del tiempo con rAF para que el resaltado sea suave
  useRafTimeSync(() => {
    const t = audioRef.current?.currentTime ?? 0;
    setCurrentTime(t);
  }, isPlaying);

  const onLoaded = () => {
    const d = audioRef.current?.duration ?? 0;
    setDuration(isFinite(d) ? d : 0);
  };

  const toggle = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  const onSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = Number(e.target.value);
    if (audioRef.current) {
      audioRef.current.currentTime = v;
    }
    setCurrentTime(v);
  };

  // Encuentra el segmento y palabra activa
  const [activeSegment, activeWord] = useMemo((): [
    KaraokeSegment | null,
    KaraokeWord | null
  ] => {
    if (!segments.length) return [null, null];
    const s =
      segments.find(s => currentTime >= s.start && currentTime < s.end) ?? null;
    if (!s) return [null, null];
    const w =
      s.words.find(w => currentTime >= w.start && currentTime < w.end) ??
      s.words.at(-1) ??
      null;
    return [s, w];
  }, [segments, currentTime]);

  // Autoscroll al segmento activo
  useEffect(() => {
    if (!scrollRef.current || !activeSegment) return;
    const el = document.getElementById(`seg-${activeSegment.start.toFixed(2)}`);
    if (el) {
      const parent = scrollRef.current;
      const top = el.offsetTop - parent.clientHeight / 2 + el.clientHeight / 2;
      parent.scrollTo({ top, behavior: 'smooth' });
    }
  }, [activeSegment]);

  const goTo = (t: number) => {
    if (!audioRef.current) return;
    audioRef.current.currentTime = t;
    setCurrentTime(t);
    if (!isPlaying) {
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  if (!audioUrl) {
    return (
      <div className="rounded-2xl border p-6 text-sm text-muted-foreground">
        Sube un archivo para habilitar el reproductor.
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
      {/* Player central */}
      <div className="lg:col-span-7">
        <div className="rounded-2xl border p-4">
          <div className="flex items-center gap-3">
            <button
              onClick={toggle}
              className="inline-flex h-10 w-10 items-center justify-center rounded-full border hover:bg-secondary"
              aria-label={isPlaying ? 'Pausa' : 'Play'}
            >
              {isPlaying ? <Pause /> : <Play />}
            </button>
            <div className="flex-1">
              <input
                type="range"
                min={0}
                max={Math.max(duration, 0)}
                step={0.01}
                value={currentTime}
                onChange={onSeek}
                className="w-full accent-foreground"
              />
              <div className="mt-1 flex justify-between text-xs tabular-nums text-muted-foreground">
                <span>{formatTime(currentTime)}</span>
                <span>{formatTime(duration)}</span>
              </div>
            </div>
          </div>

          <audio
            ref={audioRef}
            src={audioUrl}
            onLoadedMetadata={onLoaded}
            onEnded={() => setIsPlaying(false)}
            className="hidden"
            preload="metadata"
          />

          {/* Karaoke grande (palabras) */}
          <div className="mt-6 min-h-40 rounded-xl bg-secondary/30 p-4 text-lg leading-8">
            {activeSegment ? (
              <KaraokeLine segment={activeSegment} activeWord={activeWord} />
            ) : (
              <div className="text-muted-foreground">
                Reproduce para ver la letra…
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Lista de segmentos, clickable para saltar */}
      <div className="lg:col-span-5">
        <div className="rounded-2xl border">
          <div className="border-b p-3 text-sm font-semibold">
            Subtítulos por segmentos
          </div>
          <div
            ref={scrollRef}
            className="max-h-[520px] space-y-2 overflow-auto p-3"
          >
            {segments.map(s => {
              const active = activeSegment && s.start === activeSegment.start;
              return (
                <button
                  key={s.start}
                  id={`seg-${s.start.toFixed(2)}`}
                  onClick={() => goTo(s.start)}
                  className={classNames(
                    'w-full rounded-xl border p-3 text-left transition',
                    active
                      ? 'bg-primary/10 border-primary'
                      : 'hover:bg-secondary'
                  )}
                >
                  <div className="mb-1 flex items-center justify-between text-xs text-muted-foreground">
                    <span>
                      {formatTime(s.start)} - {formatTime(s.end)}
                    </span>
                  </div>
                  <div className="line-clamp-2">{s.text}</div>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

function KaraokeLine({
  segment,
  activeWord,
}: {
  segment: KaraokeSegment;
  activeWord: KaraokeWord | null;
}) {
  return (
    <div className="flex flex-wrap gap-x-1">
      {segment.words.map((w, i) => {
        const isActive =
          activeWord &&
          w.start === activeWord.start &&
          w.end === activeWord.end;
        return (
          <span
            key={i}
            className={classNames(
              'rounded px-1',
              isActive && 'bg-foreground text-background transition-colors'
            )}
          >
            {w.text}
          </span>
        );
      })}
    </div>
  );
}

function formatTime(t: number) {
  if (!isFinite(t)) return '0:00';
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}
