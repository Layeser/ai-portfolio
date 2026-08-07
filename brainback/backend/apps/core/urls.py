from django.urls import path, include
from rest_framework import routers
from . import views

router = routers.DefaultRouter()
router.register(r'users', views.UserViewSet, basename='user')
router.register(r'opportunities', views.OpportunityViewSet, basename='opportunity')
router.register(r'applications', views.ApplicationViewSet, basename='application')
router.register(r'mentorship-requests', views.MentorshipRequestViewSet, basename='mentorshiprequest')

urlpatterns = [
    # Authentication endpoints
    path('auth/register/', views.UserRegistrationView.as_view(), name='api-register'),
    path('auth/login/', views.UserLoginView.as_view(), name='api-login'),
    path('auth/logout/', views.UserLogoutView.as_view(), name='api-logout'),
    path('auth/password/change/', views.PasswordChangeView.as_view(), name='api-password-change'),
    path('auth/password/reset/', views.PasswordResetView.as_view(), name='api-password-reset'),
    path('auth/password/reset/confirm/', views.PasswordResetConfirmView.as_view(), name='api-password-reset-confirm'),
    
    # Current user profile
    path('profile/', views.UserProfileView.as_view(), name='api-profile'),
    path('profile/intention-retour/', views.IntentionRetourView.as_view(), name='api-intention-retour'),
    
    # Include router URLs
    path('', include(router.urls)),
    
    # Additional custom endpoints
    path('my-applications/', views.MyApplicationsView.as_view(), name='api-my-applications'),
    path('my-mentorship-requests/', views.MyMentorshipRequestsView.as_view(), name='api-my-mentorship-requests'),
    path('mentors/', views.MentorListView.as_view(), name='api-mentors'),
    path('opportunities/<int:pk>/apply/', views.OpportunityApplyView.as_view(), name='api-opportunity-apply'),
    path('mentorship-requests/<int:pk>/respond/', views.MentorshipRequestRespondView.as_view(), name='api-mentorship-respond'),
]