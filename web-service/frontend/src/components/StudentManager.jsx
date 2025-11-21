import { useState, useEffect } from 'react';
import api from '../api';

export default function StudentManager() {
  const [students, setStudents] = useState([]);
  const [inputText, setInputText] = useState("");
  const [loading, setLoading] = useState(false);

  const fetchStudents = async () => {
    try {
      const response = await api.get('/students/');
      setStudents(response.data);
    } catch (error) {
      console.error("학생 목록 로딩 실패:", error);
    }
  };

  useEffect(() => {
    fetchStudents();
  }, []);

  const handleUpload = async () => {
    if (!inputText.trim()) return alert("내용을 입력해주세요.");
    setLoading(true);
    try {
      const res = await api.post('/students/upload_text_batch/', {
        student_data: inputText
      });
      alert(`${res.data.count}명 등록 성공!`);
      setInputText("");
      fetchStudents();
    } catch (error) {
      alert("등록 실패: " + (error.response?.data?.error || error.message));
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm("삭제하시겠습니까?")) return;
    try {
      await api.delete(`/students/${id}/`);
      fetchStudents();
    } catch (error) {
      alert("삭제 실패");
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <h2 className="text-xl font-bold mb-4 text-gray-800">📝 학생 관리</h2>
      
      {/* 입력창 */}
      <div className="mb-6">
        <textarea
          className="w-full h-32 p-3 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 text-sm font-mono"
          placeholder={`[이름] [수강료] [교재비]\n예시:\n김철수 250000\n이영희 280000 교재비 30000`}
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
        />
        <button
          onClick={handleUpload}
          disabled={loading}
          className={`w-full mt-2 py-2 rounded text-white font-bold ${
            loading ? 'bg-gray-400' : 'bg-blue-600 hover:bg-blue-700'
          }`}
        >
          {loading ? "등록 중..." : "일괄 등록하기"}
        </button>
      </div>

      {/* 목록 */}
      <h3 className="font-semibold mb-2 text-gray-700">등록된 학생 ({students.length}명)</h3>
      <div className="overflow-y-auto max-h-[500px] border rounded">
        <table className="w-full text-sm text-left">
          <thead className="bg-gray-50 text-gray-600 sticky top-0">
            <tr>
              <th className="p-3">이름</th>
              <th className="p-3">수강료</th>
              <th className="p-3">교재비</th>
              <th className="p-3 text-center">관리</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {students.map((s) => (
              <tr key={s.id} className="hover:bg-gray-50">
                <td className="p-3 font-medium">{s.name}</td>
                <td className="p-3">{s.base_fee.toLocaleString()}</td>
                <td className="p-3 text-gray-500">{s.book_fee.toLocaleString()}</td>
                <td className="p-3 text-center">
                  <button 
                    onClick={() => handleDelete(s.id)}
                    className="text-red-500 hover:text-red-700 font-bold"
                  >
                    삭제
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}