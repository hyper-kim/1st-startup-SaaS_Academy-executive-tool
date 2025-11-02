from rest_framework import serializers
from .models import Student, Payment # 우리가 만든 모델을 가져옵니다.

class StudentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Student
        fields = '__all__'  # Student 모델의 모든 필드(name, base_fee 등)를 사용

class PaymentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Payment
        fields = '__all__'