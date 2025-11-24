import { useEffect, useMemo, useRef } from 'react';
import { useTranscribeStore } from '@/store/useTranscribeStore';

type Props = {
  onSeek?: (t: number) => void; // compatibilidad con tu App.tsx actual
};

export function SegmentList({ onSeek }: Props) {
  const data = useTranscribeStore(s => s.data);
  const currentTime = useTranscribeStore(s => s.currentTime);
  const seek = useTranscribeStore(s => s.seek);

  const listRef = useRef<HTMLDivElement>(null);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const segments = data?.segments ?? [];

  // cuál está activo
  const activeIdx = useMemo(() => {
    const i = segments.findIndex(
      s => currentTime >= s.start && currentTime < s.end
    );
    return i === -1 ? 0 : i;
  }, [segments, currentTime]);

  // autoscroll del item activo (como Spotify queue)
  useEffect(() => {
    const container = listRef.current;
    if (!container) return;
    const item = container.querySelector<HTMLDivElement>(
      `[data-seg="${activeIdx}"]`
    );
    if (!item) return;

    const cRect = container.getBoundingClientRect();
    const iRect = item.getBoundingClientRect();
    const visible = iRect.top >= cRect.top && iRect.bottom <= cRect.bottom;

    if (!visible) {
      item.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [activeIdx]);

  const handleClick = (t: number) => {
    // adelantamos 0.01s para forzar repaint de word-highlighting
    const target = Math.max(0, t + 0.01);
    if (onSeek) onSeek(target);
    else seek(target);
  };

  return (
    <div className="p-2">
      <div className="text-sm font-medium mb-2">Subtítulos por segmentos</div>
      <div
        ref={listRef}
        className="space-y-2 max-h-[520px] overflow-y-auto pr-1"
      >
        {segments.map((s, idx) => {
          const isActive = idx === activeIdx;
          const top = s.top_emotion;
          const pill = top ? `${top.label} (${top.score.toFixed(2)})` : '';

          return (
            <div
              key={idx}
              data-seg={idx}
              role="button"
              tabIndex={0}
              onClick={() => handleClick(s.start)}
              onKeyDown={e => e.key === 'Enter' && handleClick(s.start)}
              className={
                'rounded-lg border p-3 text-sm cursor-pointer focus:outline-none ' +
                (isActive ? 'bg-muted ring-1 ring-ring' : 'hover:bg-muted/50')
              }
            >
              <div className="text-[11px] opacity-70 mb-1">
                {formatTime(s.start)} - {formatTime(s.end)}
              </div>
              <div className="line-clamp-2">{s.text}</div>
              {pill && (
                <div className="mt-2 text-[11px] inline-flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded-full border">
                    {pill}
                  </span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function formatTime(t: number) {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}
