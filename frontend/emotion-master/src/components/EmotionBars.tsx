import type { EmotionScore } from '@/types/karaokeEmotion';

export function EmotionBars({ items }: { items: EmotionScore[] }) {
  if (!items?.length) return null;
  const max = Math.max(...items.map(i => i.score));
  return (
    <div className="space-y-2">
      <div className="font-medium">Emociones globales</div>
      <div className="space-y-1">
        {items.map(e => (
          <div key={e.label} className="flex items-center gap-2">
            <div className="w-24 text-sm">{e.label}</div>
            <div className="flex-1 h-2 rounded bg-muted">
              <div
                className="h-2 rounded bg-foreground/80"
                style={{ width: `${(e.score / (max || 1)) * 100}%` }}
              />
            </div>
            <div className="w-14 text-right text-xs tabular-nums">
              {e.score.toFixed(3)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
