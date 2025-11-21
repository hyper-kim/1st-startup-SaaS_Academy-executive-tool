import { useState } from 'react';
import api from '../api';

export default function ReceiptUploader() {
  const [textInput, setTextInput] = useState("");
  const [file, setFile] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async () => {
    if (!textInput && !file) return alert("텍스트나 이미지를 입력해주세요.");

    setLoading(true);
    const formData = new FormData();
    if (textInput) formData.append("text_input", textInput);
    if (file) formData.append("image_file", file);

    try {
      // 파일 전송 시엔 헤더가 자동으로 설정되게 api 인스턴스 대신 직접 설정
      const res = await api.post('/matching/upload_data/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResults(res.data.results);
    } catch (error) {
      console.error(error);
      alert("분석 실패: " + (error.response?.data?.error || "서버 오류"));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-md h-full">
      <h2 className="text-xl font-bold mb-4 text-gray-800">🧾 영수증 / 이체내역 분석</h2>

      {/* 입력 영역 */}
      <div className="space-y-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">은행 앱 텍스트 붙여넣기</label>
          <textarea
            className="w-full h-24 p-3 border border-gray-300 rounded focus:ring-2 focus:ring-green-500 text-sm"
            placeholder="원주정산 220,000..."
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">또는 영수증 사진 업로드</label>
          <input 
            type="file" 
            accept="image/*"
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-green-50 file:text-green-700 hover:file:bg-green-100"
          />
        </div>

        <button
          onClick={handleSubmit}
          disabled={loading}
          className={`w-full py-3 rounded text-white font-bold text-lg ${
            loading ? 'bg-gray-400' : 'bg-green-600 hover:bg-green-700'
          }`}
        >
          {loading ? "AI 분석 중..." : "분석 시작"}
        </button>
      </div>

      {/* 결과 영역 */}
      <div className="border-t pt-4">
        <h3 className="font-semibold mb-3 text-gray-700">분석 결과</h3>
        <div className="bg-gray-50 p-4 rounded min-h-[200px] max-h-[400px] overflow-y-auto">
          {results.length === 0 ? (
            <p className="text-gray-400 text-center mt-10">아직 분석 결과가 없습니다.</p>
          ) : (
            <ul className="space-y-2">
              {results.map((res, idx) => (
                <li key={idx} className={`p-3 rounded border ${
                  res.includes("성공") ? "bg-blue-50 border-blue-200 text-blue-800" :
                  res.includes("제안") ? "bg-yellow-50 border-yellow-200 text-yellow-800" :
                  "bg-red-50 border-red-200 text-red-800"
                }`}>
                  {res}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}