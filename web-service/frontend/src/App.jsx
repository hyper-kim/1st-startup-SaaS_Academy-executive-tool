// src/App.jsx
import { useState, useEffect } from 'react';
import api from './api'; // 우리가 만든 api 도구

function App() {
  const [students, setStudents] = useState([]);
  const [inputText, setInputText] = useState("");
  const [loading, setLoading] = useState(false);

  // 1. 학생 목록 불러오기
  const fetchStudents = async () => {
    try {
      const response = await api.get('/students/');
      setStudents(response.data);
    } catch (error) {
      console.error("학생 목록 로딩 실패:", error);
      alert("서버 연결 실패! Django 서버가 켜져 있나요?");
    }
  };

  // 페이지가 처음 뜰 때 목록 불러오기
  useEffect(() => {
    fetchStudents();
  }, []);

  // 2. 텍스트 일괄 등록
  const handleUpload = async () => {
    if (!inputText.trim()) return alert("내용을 입력해주세요.");
    
    setLoading(true);
    try {
      await api.post('/students/upload_text_batch/', {
        student_data: inputText
      });
      alert("등록 성공!");
      setInputText(""); // 입력창 비우기
      fetchStudents();  // 목록 새로고침
    } catch (error) {
      console.error("업로드 실패:", error);
      alert("등록 실패! 형식을 확인해주세요.");
    } finally {
      setLoading(false);
    }
  };

  // 3. 학생 삭제
  const handleDelete = async (id) => {
    if (!window.confirm("정말 삭제하시겠습니까?")) return;
    try {
      await api.delete(`/students/${id}/`);
      fetchStudents();
    } catch (error) {
      console.error("삭제 실패:", error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-4xl mx-auto">
        
        {/* 헤더 */}
        <header className="mb-8 text-center">
          <h1 className="text-3xl font-bold text-blue-600 mb-2">학원 정산 관리자</h1>
          <p className="text-gray-600">학생 관리 및 영수증 처리 시스템</p>
        </header>

        {/* 입력 섹션 */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4">📝 학생 일괄 등록</h2>
          <textarea
            className="w-full h-32 p-3 border rounded-md focus:ring-2 focus:ring-blue-500 focus:outline-none mb-4 font-mono text-sm"
            placeholder={`예시:\n김철수 250000\n이영희 280000 교재비 30000`}
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
          />
          <button
            onClick={handleUpload}
            disabled={loading}
            className={`w-full py-3 rounded-md text-white font-bold transition
              ${loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'}`}
          >
            {loading ? "처리 중..." : "일괄 등록하기"}
          </button>
        </div>

        {/* 목록 섹션 */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4">📋 등록된 학생 ({students.length}명)</h2>
          
          {students.length === 0 ? (
            <p className="text-center text-gray-400 py-8">등록된 학생이 없습니다.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-gray-50 border-b">
                    <th className="p-3 text-gray-600 font-medium">이름</th>
                    <th className="p-3 text-gray-600 font-medium">수강료</th>
                    <th className="p-3 text-gray-600 font-medium">교재비</th>
                    <th className="p-3 text-gray-600 font-medium">비고</th>
                    <th className="p-3 text-gray-600 font-medium">관리</th>
                  </tr>
                </thead>
                <tbody>
                  {students.map((student) => (
                    <tr key={student.id} className="border-b hover:bg-gray-50">
                      <td className="p-3 font-medium">{student.name}</td>
                      <td className="p-3">{student.base_fee.toLocaleString()}원</td>
                      <td className="p-3 text-gray-500">{student.book_fee.toLocaleString()}원</td>
                      <td className="p-3 text-sm text-gray-400">{student.notes}</td>
                      <td className="p-3">
                        <button
                          onClick={() => handleDelete(student.id)}
                          className="text-red-500 hover:text-red-700 text-sm font-bold"
                        >
                          삭제
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}

export default App;