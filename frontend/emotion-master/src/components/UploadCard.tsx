import { api } from '@/api/client';
import { usePlayerStore } from '@/store/usePlayerStore';
import type { KaraokeResponse } from '@/types';
import { useMutation } from '@tanstack/react-query';
import { useRef, useState } from 'react';

export default function UploadCard() {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [fileName, setFileName] = useState<string>('');
  const { setAudioUrl, loadKaraoke, reset } = usePlayerStore();

  const mutation = useMutation({
    mutationFn: async (file: File) => {
      const fd = new FormData();
      fd.append('file', file);
      const { data } = await api.post<KaraokeResponse>(
        '/transcribe/karaoke',
        fd,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
        }
      );
      return data;
    },
    onSuccess: data => {
      loadKaraoke(data);
    },
  });

  const onPick = () => inputRef.current?.click();

  const onChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFileName(f.name);
    // Url local para reproducir el mismo archivo que enviamos
    const localUrl = URL.createObjectURL(f);
    setAudioUrl(localUrl);
    // reiniciar estado karaoke y hacer la llamada
    mutation.reset();
    reset();
    setAudioUrl(localUrl);
    setFileName(f.name);
    mutation.mutate(f);
  };

  return (
    <div className="rounded-2xl border p-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">Subir audio y transcribir</h3>
          <p className="text-sm text-muted-foreground">
            Envía un archivo de audio. Obtendrás transcripción y palabras con
            timestamp.
          </p>
        </div>
        <button
          onClick={onPick}
          className="rounded-xl border px-4 py-2 hover:bg-secondary"
        >
          Elegir archivo
        </button>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept="audio/*"
        onChange={onChange}
        className="hidden"
      />

      {fileName ? (
        <p className="mt-2 text-sm">
          Archivo: <span className="font-medium">{fileName}</span>
        </p>
      ) : null}

      <div className="mt-3 text-sm">
        {mutation.isPending && (
          <span className="text-amber-600">Procesando…</span>
        )}
        {mutation.isError && (
          <span className="text-red-600">
            Error:{' '}
            {(mutation.error as Error)?.message ?? 'falló la transcripción'}
          </span>
        )}
        {mutation.isSuccess && (
          <span className="text-emerald-600">Listo ✅</span>
        )}
      </div>
    </div>
  );
}
