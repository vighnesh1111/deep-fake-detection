"""project_settings URL Configuration
"""
from django.contrib import admin
from django.urls import path, include
from . import views
from .views import about, index, predict_page,cuda_full,analyze_video_youtube

app_name = 'ml_app'
handler404 = views.handler404

urlpatterns = [
    path('', index, name='home'),
    path('about/', about, name='about'),
    path('predict/', predict_page, name='predict'),
    path('cuda_full/',cuda_full,name='cuda_full'),
    path('image_index/', views.image_index, name='image_index'),  # Add this line
    path('image_predict/', views.image_predict, name='image_predict'),  # Add this line
    path('analyze_youtube/', views.analyze_video_youtube, name='analyze_video_youtube'),  # Add this line  
]
