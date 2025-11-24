import { useEffect, useRef } from 'react';
import { usePlayerStore } from '@/store/player';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';

function fmt(t: number) {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export function PlayerBar() {
  const audioUrl = usePlayerStore(s => s.audioUrl);
  const isPlaying = usePlayerStore(s => s.isPlaying);
  const setPlaying = usePlayerStore(s => s.setPlaying);
  const currentTime = usePlayerStore(s => s.currentTime);
  const setTime = usePlayerStore(s => s.setTime);
  const duration = usePlayerStore(s => s.duration);
  const setDuration = usePlayerStore(s => s.setDuration);

  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;
    const onTime = () => setTime(el.currentTime);
    const onMeta = () => setDuration(el.duration || 0);
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    el.addEventListener('timeupdate', onTime);
    el.addEventListener('loadedmetadata', onMeta);
    el.addEventListener('play', onPlay);
    el.addEventListener('pause', onPause);
    return () => {
      el.removeEventListener('timeupdate', onTime);
      el.removeEventListener('loadedmetadata', onMeta);
      el.removeEventListener('play', onPlay);
      el.removeEventListener('pause', onPause);
    };
  }, [setTime, setDuration, setPlaying]);

  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;
    if (isPlaying) el.play().catch(() => {});
    else el.pause();
  }, [isPlaying]);

  if (!audioUrl) return null;

  const pct = duration > 0 ? (currentTime / duration) * 100 : 0;

  const onSeek = (vals: number[]) => {
    const t = ((vals?.[0] ?? 0) / 100) * duration;
    const el = audioRef.current;
    if (el) el.currentTime = t;
    setTime(t);
  };

  return (
    <Card className="border-border/60">
      <CardHeader>
        <CardTitle>Reproductor</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <audio ref={audioRef} src={audioUrl} preload="metadata" />
        <div className="flex items-center gap-2">
          <Button variant="secondary" onClick={() => setPlaying(!isPlaying)}>
            {isPlaying ? '⏸ Pausa' : '▶️ Reproducir'}
          </Button>
          <div className="text-xs text-muted-foreground">
            {fmt(currentTime)} / {fmt(duration || 0)}
          </div>
        </div>
        <Slider value={[pct]} onValueChange={onSeek} />
      </CardContent>
    </Card>
  );
}
