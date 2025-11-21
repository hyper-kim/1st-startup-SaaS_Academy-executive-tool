import StudentManager from './components/StudentManager';
import ReceiptUploader from './components/ReceiptUploader';

function App() {
  return (
    <div className="min-h-screen bg-gray-100">
      {/* 상단 네비게이션 */}
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <h1 className="text-2xl font-bold text-gray-900">🎓 학원 정산 비서</h1>
            <span className="text-sm text-gray-500">AI Powered System</span>
          </div>
        </div>
      </nav>

      {/* 메인 컨텐츠 */}
      <main className="max-w-7xl mx-auto py-10 px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* 왼쪽: 영수증 처리 (모바일에서 위로 옴) */}
          <div className="order-1 lg:order-1">
            <ReceiptUploader />
          </div>
          
          {/* 오른쪽: 학생 관리 */}
          <div className="order-2 lg:order-2">
            <StudentManager />
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;