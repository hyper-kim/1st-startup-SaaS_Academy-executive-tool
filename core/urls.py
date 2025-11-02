# core/urls.py (새 파일)

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views # core/views.py 파일을 가져옵니다.

# 라우터를 생성합니다.
router = DefaultRouter()

# 라우터에 ViewSet을 등록합니다.
# 'students'라는 URL 경로에 StudentViewSet을 연결합니다.
router.register(r'students', views.StudentViewSet)
router.register(r'payments', views.PaymentViewSet)
router.register(r'matching', views.MatchingViewSet, basename='matching')

# 이제 라우터가 자동으로 URL 패턴을 생성해줍니다.
# (예: /students/, /students/1/ 등)
urlpatterns = [
    path('', include(router.urls)),
]