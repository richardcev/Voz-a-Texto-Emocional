import axios from 'axios';
import type { KaraokeEmotionResponse } from '@/types/karaokeEmotion';

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:7777';

export async function postKaraokeEmotionEsMaster(file: File) {
  const fd = new FormData();
  fd.append('file', file);
  const { data } = await axios.post<KaraokeEmotionResponse>(
    `${API_BASE}/transcribe/karaoke-emotion-es-master`,
    fd,
    { headers: { 'Content-Type': 'multipart/form-data' } }
  );
  return data;
}
