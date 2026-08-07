from rest_framework import serializers
from django.contrib.auth import authenticate
from .models import User, MentorshipRequest, Opportunity, Application
import os
from django.utils.timezone import now


class UserRegistrationSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=8)
    password_confirm = serializers.CharField(write_only=True)
    
    class Meta:
        model = User
        fields = [
            'id', 'first_name', 'last_name', 'email', 'password', 'password_confirm',
            'user_type', 'boursier_status', 'promotion_cpge', 'cpge_etablissement',
            'pays_cpge', 'pays_actuel', 'filiere_cpge', 'etablissement_etudes', 
            'specialite', 'niveau_etudes', 'annee_integration_cycle_ingenieur', 
            'fin_cycle_ingenieur', 'date_fin_etudes', 'entreprise_actuelle',
            'poste_actuel', 'secteur_activite', 'intention_retour',
            'conditions_retour', 'besoins_accompagnement', 'supervisor_type',
            'institution_supervision', 'pays_cycle_ingenieur', 'pays_mobilite', 
            'pays_post_ingenieur', 'ecoles_mobilite', 'ecoles_post_ingenieur', 
            'date_debut_mobilite', 'date_fin_mobilite', 'date_debut_post_ingenieur', 
            'date_fin_post_ingenieur'
        ]
        extra_kwargs = {
            'password': {'write_only': True},
            'password_confirm': {'write_only': True},
        }

    def validate(self, data):
        if data.get('password') != data.get('password_confirm'):
            raise serializers.ValidationError({"password_confirm": "Passwords do not match."})
        
        user_type = data.get('user_type')
        
        # Validate user type specific fields
        if user_type == 'boursier' and not data.get('boursier_status'):
            raise serializers.ValidationError({"boursier_status": "This field is required for boursiers."})
        
        if user_type == 'supervisor' and not data.get('supervisor_type'):
            raise serializers.ValidationError({"supervisor_type": "This field is required for supervisors."})
        
        return data

    def create(self, validated_data):
        validated_data.pop('password_confirm')
        password = validated_data.pop('password')
        
        user = User(**validated_data)
        user.set_password(password)
        user.username = validated_data['email']  # Use email as username
        user.save()
        
        return user


class UserLoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField()

    def validate(self, data):
        email = data.get('email')
        password = data.get('password')

        if email and password:
            user = authenticate(username=email, password=password)
            
            if not user:
                raise serializers.ValidationError("Invalid email or password.")
            
            if not user.is_approved:
                raise serializers.ValidationError("Your account is not yet approved by the administrator.")
            
            if not user.is_active:
                raise serializers.ValidationError("Your account is deactivated.")
            
            data['user'] = user
            return data
        
        raise serializers.ValidationError("Email and password are required.")


class UserProfileSerializer(serializers.ModelSerializer):
    profile_photo_url = serializers.SerializerMethodField()
    cv_file_url = serializers.SerializerMethodField()
    full_name = serializers.ReadOnlyField()
    is_etudiant_en_cours = serializers.ReadOnlyField()
    is_diplome = serializers.ReadOnlyField()
    display_promotion = serializers.ReadOnlyField()

    class Meta:
        model = User
        fields = [
            'id', 'first_name', 'last_name', 'email', 'full_name', 'user_type',
            'profile_photo', 'profile_photo_url', 'linkedin_url', 'cv_file', 'cv_file_url',
            'boursier_status', 'promotion_cpge', 'cpge_etablissement', 'pays_cpge',
            'pays_actuel', 'filiere_cpge', 'etablissement_etudes', 'specialite',
            'niveau_etudes', 'annee_integration_cycle_ingenieur', 'fin_cycle_ingenieur',
            'date_fin_etudes', 'entreprise_actuelle', 'poste_actuel', 'secteur_activite',
            'intention_retour', 'conditions_retour', 'besoins_accompagnement',
            'supervisor_type', 'institution_supervision', 'is_mentor', 'is_approved',
            'is_alumni_member', 'is_etudiant_en_cours', 'is_diplome', 
            'display_promotion', 'pays_cycle_ingenieur', 'pays_mobilite', 
            'pays_post_ingenieur', 'ecoles_mobilite', 'ecoles_post_ingenieur',
            'date_debut_mobilite', 'date_fin_mobilite', 'date_debut_post_ingenieur',
            'date_fin_post_ingenieur', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'email', 'is_approved', 'is_alumni_member', 'created_at', 'updated_at']

    def get_profile_photo_url(self, obj):
        if obj.profile_photo:
            return obj.profile_photo.url
        return None

    def get_cv_file_url(self, obj):
        if obj.cv_file:
            return obj.cv_file.url
        return None

    def validate_profile_photo(self, value):
        if value:
            if value.size > 2 * 1024 * 1024:  # 2MB
                raise serializers.ValidationError("Profile photo size should not exceed 2MB.")
            if not value.name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                raise serializers.ValidationError("Only image files are allowed.")
        return value

    def validate_cv_file(self, value):
        if value:
            if value.size > 5 * 1024 * 1024:  # 5MB
                raise serializers.ValidationError("CV file size should not exceed 5MB.")
            if not value.name.lower().endswith('.pdf'):
                raise serializers.ValidationError("Only PDF files are allowed for CV.")
        return value


class IntentionRetourSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = [
            'intention_retour', 'conditions_retour', 'besoins_accompagnement',
            'secteur_prioritaire', 'support_networking', 'support_housing',
            'support_legal', 'support_career'
        ]


class MentorshipRequestSerializer(serializers.ModelSerializer):
    student_name = serializers.CharField(source='student.full_name', read_only=True)
    supervisor_name = serializers.CharField(source='supervisor.full_name', read_only=True)
    student_email = serializers.EmailField(source='student.email', read_only=True)

    class Meta:
        model = MentorshipRequest
        fields = [
            'id', 'student', 'student_name', 'student_email', 'supervisor', 
            'supervisor_name', 'status', 'message', 'created_at', 'responded_at'
        ]
        read_only_fields = ['id', 'created_at', 'responded_at']

    def validate(self, data):
        student = data.get('student')
        supervisor = data.get('supervisor')

        if student and supervisor:
            if student.user_type != 'boursier':
                raise serializers.ValidationError({"student": "Only boursiers can request mentorship."})
            
            if not supervisor.is_mentor:
                raise serializers.ValidationError({"supervisor": "This user is not available as a mentor."})
            
            # Check if request already exists
            existing_request = MentorshipRequest.objects.filter(
                student=student, supervisor=supervisor
            ).exists()
            
            if existing_request:
                raise serializers.ValidationError("A mentorship request already exists for this student and supervisor.")
        
        return data


class OpportunitySerializer(serializers.ModelSerializer):
    supervisor_name = serializers.CharField(source='supervisor.full_name', read_only=True)
    company = serializers.CharField(max_length=100)
    application_count = serializers.SerializerMethodField()

    class Meta:
        model = Opportunity
        fields = [
            'id', 'title', 'company', 'location', 'type', 'contract_type',
            'experience_level', 'description', 'requirements', 'responsibilities',
            'salary_min', 'salary_max', 'salary_currency', 'application_deadline',
            'start_date', 'supervisor', 'supervisor_name', 'is_active', 
            'created_at', 'updated_at', 'application_count'
        ]
        read_only_fields = ['id', 'supervisor_name', 'created_at', 'updated_at', 'application_count']

    def get_application_count(self, obj):
        return obj.applications.count()

    def validate_application_deadline(self, value):
        if value < now().date():
            raise serializers.ValidationError("Application deadline cannot be in the past.")
        return value

    def validate_salary(self, data):
        salary_min = data.get('salary_min')
        salary_max = data.get('salary_max')
        
        if salary_min and salary_max and salary_min > salary_max:
            raise serializers.ValidationError("Minimum salary cannot be greater than maximum salary.")
        
        return data


class ApplicationSerializer(serializers.ModelSerializer):
    student_name = serializers.CharField(source='student.full_name', read_only=True)
    student_email = serializers.EmailField(source='student.email', read_only=True)
    opportunity_title = serializers.CharField(source='opportunity.title', read_only=True)
    company_name = serializers.CharField(source='opportunity.company', read_only=True)
    cv_file_url = serializers.SerializerMethodField()
    additional_documents_url = serializers.SerializerMethodField()

    class Meta:
        model = Application
        fields = [
            'id', 'opportunity', 'opportunity_title', 'student', 'student_name',
            'student_email', 'company_name', 'status', 'cover_letter', 'cv_file',
            'cv_file_url', 'additional_documents', 'additional_documents_url',
            'supervisor_notes', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'student_name', 'student_email', 'opportunity_title', 
            'company_name', 'cv_file_url', 'additional_documents_url', 
            'created_at', 'updated_at', 'supervisor_notes'
        ]

    def get_cv_file_url(self, obj):
        if obj.cv_file:
            return obj.cv_file.url
        return None

    def get_additional_documents_url(self, obj):
        if obj.additional_documents:
            return obj.additional_documents.url
        return None

    def validate_cv_file(self, value):
        if value:
            if not value.name.lower().endswith('.pdf'):
                raise serializers.ValidationError("CV must be in PDF format.")
            if value.size > 5 * 1024 * 1024:  # 5MB
                raise serializers.ValidationError("CV file size should not exceed 5MB.")
        return value

    def validate_additional_documents(self, value):
        if value:
            allowed_extensions = ['.pdf', '.doc', '.docx']
            ext = os.path.splitext(value.name)[1].lower()
            if ext not in allowed_extensions:
                raise serializers.ValidationError("Unsupported file format. Use PDF, DOC or DOCX.")
            if value.size > 10 * 1024 * 1024:  # 10MB
                raise serializers.ValidationError("Additional documents should not exceed 10MB.")
        return value

    def validate(self, data):
        student = self.context['request'].user
        opportunity = data.get('opportunity')
        
        if opportunity and student:
            # Check if student has already applied
            existing_application = Application.objects.filter(
                opportunity=opportunity, student=student
            ).exists()
            
            if existing_application:
                raise serializers.ValidationError("You have already applied to this opportunity.")
            
            # Check if opportunity is still active
            if not opportunity.is_active:
                raise serializers.ValidationError("This opportunity is no longer active.")
            
            # Check application deadline
            from django.utils.timezone import now
            if opportunity.application_deadline < now().date():
                raise serializers.ValidationError("The application deadline for this opportunity has passed.")
        
        return data


class ApplicationStatusSerializer(serializers.ModelSerializer):
    class Meta:
        model = Application
        fields = ['status', 'supervisor_notes']
        read_only_fields = ['supervisor_notes']


class UserListSerializer(serializers.ModelSerializer):
    full_name = serializers.ReadOnlyField()
    user_type_display = serializers.CharField(source='get_user_type_display', read_only=True)

    class Meta:
        model = User
        fields = [
            'id', 'full_name', 'email', 'user_type', 'user_type_display',
            'boursier_status', 'promotion_cpge', 'is_approved', 'is_active',
            'created_at'
        ]


class PasswordChangeSerializer(serializers.Serializer):
    old_password = serializers.CharField(required=True)
    new_password = serializers.CharField(required=True, min_length=8)
    confirm_password = serializers.CharField(required=True)

    def validate(self, data):
        if data['new_password'] != data['confirm_password']:
            raise serializers.ValidationError({"confirm_password": "New passwords do not match."})
        return data


class PasswordResetSerializer(serializers.Serializer):
    email = serializers.EmailField(required=True)


class PasswordResetConfirmSerializer(serializers.Serializer):
    new_password = serializers.CharField(required=True, min_length=8)
    confirm_password = serializers.CharField(required=True)

    def validate(self, data):
        if data['new_password'] != data['confirm_password']:
            raise serializers.ValidationError({"confirm_password": "Passwords do not match."})
        return data