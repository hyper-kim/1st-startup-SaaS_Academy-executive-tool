# web-service/core/urls.py

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# Router 설정
router = DefaultRouter()
router.register(r'students', views.StudentViewSet)      # /api/students/
router.register(r'payments', views.PaymentViewSet)      # /api/payments/
router.register(r'matching', views.MatchingViewSet, basename='matching') # /api/matching/

urlpatterns = [
    # 라우터가 생성한 URL 패턴 포함
    path('', include(router.urls)),
]