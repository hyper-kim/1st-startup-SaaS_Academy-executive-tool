from django.db import models

# Create your models here.
class Student(models.Model):
    name = models.CharField(max_length=100)
    parent_contact = models.CharField(max_length=20, blank=True) # 학부모 연락처
    base_fee = models.IntegerField(default=0) # 기본 월 수강료 (AI 매칭 기준)
    book_fee = models.IntegerField(default=0) # 교재비
    notes = models.TextField(blank=True) # 기타 메모
    
    def __str__(self):
        return self.name

class Payment(models.Model):
    PAYMENT_STATUS_CHOICES = [
        ('UNPAID', '미납'),
        ('PAID', '납부 완료'),
        ('MISMATCH', '금액 불일치'), # AI 매칭시 활용
    ]
    
    # 학생 모델과 1:N 관계로 연결합니다.
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='payments')
    amount_paid = models.IntegerField() # 실제 입금액
    payment_date = models.DateField() # 결제일
    payment_method = models.CharField(max_length=50, blank=True) # 예: '카드', '이체'
    status = models.CharField(max_length=10, choices=PAYMENT_STATUS_CHOICES, default='UNPAID')

    def __str__(self):
        return f"{self.student.name} - {self.amount_paid}원"