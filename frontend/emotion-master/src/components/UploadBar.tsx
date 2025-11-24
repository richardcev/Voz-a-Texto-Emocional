import { useRef } from 'react';
import { useTranscribeStore } from '@/store/useTranscribeStore';
import { postKaraokeEmotionEsMaster } from '@/lib/api';
import { Button } from '@/components/ui/button';

export function UploadBar() {
  const fileRef = useRef<HTMLInputElement>(null);
  const { setAudioFile, setAudioUrl, setData } = useTranscribeStore();

  const onPick = () => fileRef.current?.click();

  const onChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setAudioFile(f);
    const url = URL.createObjectURL(f);
    setAudioUrl(url);
    const data = await postKaraokeEmotionEsMaster(f);
    setData(data);
  };

  return (
    <div className="flex items-center justify-between gap-4">
      <div>
        <div className="font-medium">Subir audio y transcribir</div>
        <p className="text-sm text-muted-foreground">
          Envía un archivo de audio. Obtendrás transcripción, palabras y
          emociones por segmento.
        </p>
      </div>
      <div>
        <input
          ref={fileRef}
          type="file"
          accept="audio/*"
          className="hidden"
          onChange={onChange}
        />
        <Button onClick={onPick}>Elegir archivo</Button>
      </div>
    </div>
  );
}
