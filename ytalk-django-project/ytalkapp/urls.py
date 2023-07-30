from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("post_link/", views.post_link, name="post_link"), 
]
