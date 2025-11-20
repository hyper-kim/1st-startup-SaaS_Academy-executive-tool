from django.contrib import admin

# Register your models here.
from .models import Student, Payment # 방금 만든 모델 가져오기

# 관리자 사이트에 모델을 등록
admin.site.register(Student)
admin.site.register(Payment)