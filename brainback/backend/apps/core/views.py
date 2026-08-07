from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from django.contrib.auth import login, logout
from django.shortcuts import get_object_or_404
from .models import User, MentorshipRequest, Opportunity, Application
from .serializers import (
    UserRegistrationSerializer, UserLoginSerializer, UserProfileSerializer,
    IntentionRetourSerializer, MentorshipRequestSerializer, OpportunitySerializer,
    ApplicationSerializer, ApplicationStatusSerializer, UserListSerializer,
    PasswordChangeSerializer, PasswordResetSerializer, PasswordResetConfirmSerializer
)


class UserRegistrationView(APIView):
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        serializer = UserRegistrationSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            return Response({
                'message': 'User registered successfully. Waiting for approval.',
                'user_id': user.id
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class UserLoginView(APIView):
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        serializer = UserLoginSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.validated_data['user']
            login(request, user)
            return Response({
                'message': 'Login successful',
                'user': UserProfileSerializer(user).data
            })
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class UserLogoutView(APIView):
    def post(self, request):
        logout(request)
        return Response({'message': 'Logout successful'})


class UserProfileView(APIView):
    def get(self, request):
        serializer = UserProfileSerializer(request.user)
        return Response(serializer.data)
    
    def put(self, request):
        serializer = UserProfileSerializer(request.user, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class IntentionRetourView(APIView):
    def get(self, request):
        serializer = IntentionRetourSerializer(request.user)
        return Response(serializer.data)
    
    def put(self, request):
        serializer = IntentionRetourSerializer(request.user, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class PasswordChangeView(APIView):
    def post(self, request):
        serializer = PasswordChangeSerializer(data=request.data)
        if serializer.is_valid():
            if not request.user.check_password(serializer.validated_data['old_password']):
                return Response(
                    {'old_password': ['Wrong password.']}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            request.user.set_password(serializer.validated_data['new_password'])
            request.user.save()
            return Response({'message': 'Password updated successfully'})
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class PasswordResetView(APIView):
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        serializer = PasswordResetSerializer(data=request.data)
        if serializer.is_valid():
            # In a real implementation, we would send a reset email here
            return Response({'message': 'Password reset email sent'})
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class PasswordResetConfirmView(APIView):
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        serializer = PasswordResetConfirmSerializer(data=request.data)
        if serializer.is_valid():
            # In a real implementation, we would validate the token and reset password
            return Response({'message': 'Password reset successful'})
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserListSerializer
    permission_classes = [permissions.IsAdminUser]
    
    def get_serializer_class(self):
        if self.action == 'retrieve':
            return UserProfileSerializer
        return super().get_serializer_class()
    
    @action(detail=False, methods=['get'])
    def boursiers(self, request):
        boursiers = User.objects.filter(user_type=User.UserType.BOURSIER)
        serializer = self.get_serializer(boursiers, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def supervisors(self, request):
        supervisors = User.objects.filter(user_type=User.UserType.SUPERVISOR)
        serializer = self.get_serializer(supervisors, many=True)
        return Response(serializer.data)


class OpportunityViewSet(viewsets.ModelViewSet):
    queryset = Opportunity.objects.all()
    serializer_class = OpportunitySerializer
    
    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            return [permissions.IsAuthenticated(), permissions.IsAdminUser()]
        return [permissions.IsAuthenticated()]
    
    def perform_create(self, serializer):
        serializer.save(supervisor=self.request.user)
    
    @action(detail=True, methods=['get'])
    def applications(self, request, pk=None):
        opportunity = self.get_object()
        if request.user != opportunity.supervisor and not request.user.is_admin:
            return Response(
                {'error': 'You do not have permission to view these applications'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        applications = opportunity.applications.all()
        serializer = ApplicationSerializer(applications, many=True)
        return Response(serializer.data)


class ApplicationViewSet(viewsets.ModelViewSet):
    queryset = Application.objects.all()
    serializer_class = ApplicationSerializer
    
    def get_permissions(self):
        if self.action in ['update', 'partial_update', 'destroy']:
            return [permissions.IsAuthenticated(), permissions.IsAdminUser()]
        return [permissions.IsAuthenticated()]
    
    def get_queryset(self):
        if self.request.user.is_admin:
            return Application.objects.all()
        elif self.request.user.is_supervisor:
            return Application.objects.filter(opportunity__supervisor=self.request.user)
        else:
            return Application.objects.filter(student=self.request.user)


class MentorshipRequestViewSet(viewsets.ModelViewSet):
    queryset = MentorshipRequest.objects.all()
    serializer_class = MentorshipRequestSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def perform_create(self, serializer):
        serializer.save(student=self.request.user)


class MyApplicationsView(APIView):
    def get(self, request):
        applications = Application.objects.filter(student=request.user)
        serializer = ApplicationSerializer(applications, many=True)
        return Response(serializer.data)


class MyMentorshipRequestsView(APIView):
    def get(self, request):
        if request.user.is_boursier:
            requests = MentorshipRequest.objects.filter(student=request.user)
        else:
            requests = MentorshipRequest.objects.filter(supervisor=request.user)
        
        serializer = MentorshipRequestSerializer(requests, many=True)
        return Response(serializer.data)


class MentorListView(APIView):
    def get(self, request):
        mentors = User.objects.filter(is_mentor=True)
        serializer = UserListSerializer(mentors, many=True)
        return Response(serializer.data)


class OpportunityApplyView(APIView):
    def post(self, request, pk):
        opportunity = get_object_or_404(Opportunity, pk=pk, is_active=True)
        
        # Check if user has already applied
        if Application.objects.filter(opportunity=opportunity, student=request.user).exists():
            return Response(
                {'error': 'You have already applied to this opportunity'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        serializer = ApplicationSerializer(
            data=request.data,
            context={'request': request}
        )
        
        if serializer.is_valid():
            application = serializer.save(
                opportunity=opportunity,
                student=request.user
            )
            return Response(
                ApplicationSerializer(application).data,
                status=status.HTTP_201_CREATED
            )
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class MentorshipRequestRespondView(APIView):
    def post(self, request, pk):
        mentorship_request = get_object_or_404(
            MentorshipRequest, 
            pk=pk, 
            supervisor=request.user
        )
        
        if mentorship_request.status != 'PENDING':
            return Response(
                {'error': 'This request has already been responded to'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        status_value = request.data.get('status')
        if status_value not in ['ACCEPTED', 'REJECTED']:
            return Response(
                {'error': 'Status must be either ACCEPTED or REJECTED'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        mentorship_request.status = status_value
        mentorship_request.save()
        
        return Response(MentorshipRequestSerializer(mentorship_request).data)