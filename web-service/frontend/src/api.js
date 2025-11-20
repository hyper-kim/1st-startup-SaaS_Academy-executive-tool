import axios from 'axios';

const api = axios.create({
  // Vite의 Proxy 설정 덕분에 '/api'만 써도 Django(8000번)로 연결됩니다.
  baseURL: '/api', 
  headers: {
    'Content-Type': 'application/json',
  },
});

export default api;