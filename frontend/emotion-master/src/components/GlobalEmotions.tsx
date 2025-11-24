import { usePlayerStore } from '@/store/player';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function GlobalEmotions() {
  const global = usePlayerStore(s => s.globalEmotions);
  if (!global?.length) return null;

  const max = Math.max(...global.map(x => x.score), 1e-6);

  return (
    <Card className="border-border/60">
      <CardHeader>
        <CardTitle>Emociones globales</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {global.map(e => (
          <div
            key={e.label}
            className="grid grid-cols-[120px_1fr_60px] items-center gap-2"
          >
            <div className="capitalize text-sm">{e.label}</div>
            <div className="h-2 w-full rounded-full bg-muted">
              <div
                className="h-2 rounded-full bg-primary transition-all"
                style={{ width: `${(e.score / max) * 100}%` }}
              />
            </div>
            <div className="text-right text-xs text-muted-foreground">
              {e.score.toFixed(3)}
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
