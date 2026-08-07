from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.utils.translation import gettext_lazy as _
from .models import User, MentorshipRequest, Opportunity, Application


@admin.register(User)
class CustomUserAdmin(UserAdmin):
    list_display = ('email', 'first_name', 'last_name', 'user_type', 'boursier_status', 'promotion_cpge', 'is_approved', 'is_active', 'created_at')
    list_filter = ('user_type', 'boursier_status', 'supervisor_type', 'is_approved', 'is_active', 'is_alumni_member', 'created_at')
    search_fields = ('email', 'first_name', 'last_name', 'promotion_cpge', 'specialite')
    ordering = ('-created_at',)
    
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        (_('Informations personnelles'), {'fields': ('first_name', 'last_name', 'profile_photo')}),
        (_('Type d\'utilisateur'), {'fields': ('user_type', 'boursier_status', 'supervisor_type')}),
        (_('Informations CPGE'), {
            'fields': ('promotion_cpge', 'cpge_etablissement', 'filiere_cpge', 'pays_cpge'),
            'classes': ('collapse',)
        }),
        (_('Informations d\'études'), {
            'fields': (
                'etablissement_etudes', 'specialite', 'niveau_etudes', 
                'annee_integration_cycle_ingenieur', 'fin_cycle_ingenieur', 'date_fin_etudes',
                'pays_cycle_ingenieur'
            ),
            'classes': ('collapse',)
        }),
        (_('Mobilité et études avancées'), {
            'fields': (
                'ecoles_mobilite', 'date_debut_mobilite', 'date_fin_mobilite', 'pays_mobilite',
                'ecoles_post_ingenieur', 'date_debut_post_ingenieur', 'date_fin_post_ingenieur', 'pays_post_ingenieur'
            ),
            'classes': ('collapse',)
        }),
        (_('Informations professionnelles'), {
            'fields': ('entreprise_actuelle', 'poste_actuel', 'secteur_activite'),
            'classes': ('collapse',)
        }),
        (_('Intention de retour'), {
            'fields': (
                'intention_retour', 'conditions_retour', 'besoins_accompagnement',
                'secteur_prioritaire', 'support_networking', 'support_housing', 
                'support_legal', 'support_career'
            ),
            'classes': ('collapse',)
        }),
        (_('Supervision'), {
            'fields': ('institution_supervision',),
            'classes': ('collapse',)
        }),
        (_('Réseaux et documents'), {
            'fields': ('linkedin_url', 'cv_file', 'is_chercheur', 'is_mentor'),
            'classes': ('collapse',)
        }),
        (_('Statut'), {
            'fields': (
                'is_active', 'is_approved', 'is_alumni_member', 'is_staff', 
                'is_superuser', 'groups', 'user_permissions'
            )
        }),
        (_('Dates importantes'), {
            'fields': ('last_login', 'date_joined', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'first_name', 'last_name', 'user_type', 'password1', 'password2'),
        }),
    )
    
    readonly_fields = ('last_login', 'date_joined', 'created_at', 'updated_at')
    
    actions = ['approve_users', 'reject_users', 'make_alumni_members', 'remove_alumni_members']
    
    def approve_users(self, request, queryset):
        updated = queryset.update(is_approved=True, is_active=True)
        self.message_user(request, f'{updated} utilisateur(s) approuvé(s) avec succès.')
    approve_users.short_description = "Approuver les utilisateurs sélectionnés"
    
    def reject_users(self, request, queryset):
        count = queryset.count()
        queryset.delete()
        self.message_user(request, f'{count} utilisateur(s) supprimé(s) avec succès.')
    reject_users.short_description = "Rejeter les utilisateurs sélectionnés"
    
    def make_alumni_members(self, request, queryset):
        updated = queryset.filter(user_type='boursier').update(is_alumni_member=True)
        self.message_user(request, f'{updated} boursier(s) ajouté(s) au réseau Alumni.')
    make_alumni_members.short_description = "Ajouter au réseau Alumni"
    
    def remove_alumni_members(self, request, queryset):
        updated = queryset.filter(user_type='boursier').update(is_alumni_member=False)
        self.message_user(request, f'{updated} boursier(s) retiré(s) du réseau Alumni.')
    remove_alumni_members.short_description = "Retirer du réseau Alumni"
    
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        if request.user.is_superuser:
            return qs
        return qs.filter(is_superuser=False)
    
    def get_list_display(self, request):
        list_display = list(super().get_list_display(request))
        if not request.user.is_superuser:
            # Masquer certains champs pour les non-superusers
            if 'is_superuser' in list_display:
                list_display.remove('is_superuser')
        return list_display


@admin.register(MentorshipRequest)
class MentorshipRequestAdmin(admin.ModelAdmin):
    list_display = ('student', 'supervisor', 'status', 'created_at', 'responded_at')
    list_filter = ('status', 'created_at')
    search_fields = ('student__email', 'student__first_name', 'student__last_name', 
                    'supervisor__email', 'supervisor__first_name', 'supervisor__last_name')
    readonly_fields = ('created_at', 'responded_at')
    actions = ['mark_accepted', 'mark_rejected', 'mark_cancelled']
    
    def mark_accepted(self, request, queryset):
        updated = queryset.update(status='ACCEPTED')
        self.message_user(request, f'{updated} demande(s) de mentorat acceptée(s).')
    mark_accepted.short_description = "Marquer comme acceptées"
    
    def mark_rejected(self, request, queryset):
        updated = queryset.update(status='REJECTED')
        self.message_user(request, f'{updated} demande(s) de mentorat rejetée(s).')
    mark_rejected.short_description = "Marquer comme rejetées"
    
    def mark_cancelled(self, request, queryset):
        updated = queryset.update(status='CANCELLED')
        self.message_user(request, f'{updated} demande(s) de mentorat annulée(s).')
    mark_cancelled.short_description = "Marquer comme annulées"


@admin.register(Opportunity)
class OpportunityAdmin(admin.ModelAdmin):
    list_display = ('title', 'company', 'type', 'contract_type', 'is_active', 'application_deadline', 'supervisor')
    list_filter = ('type', 'contract_type', 'experience_level', 'is_active', 'created_at')
    search_fields = ('title', 'company', 'location', 'supervisor__email', 'supervisor__first_name', 'supervisor__last_name')
    readonly_fields = ('created_at', 'updated_at')
    actions = ['activate_opportunities', 'deactivate_opportunities']
    
    def activate_opportunities(self, request, queryset):
        updated = queryset.update(is_active=True)
        self.message_user(request, f'{updated} opportunité(s) activée(s).')
    activate_opportunities.short_description = "Activer les opportunités"
    
    def deactivate_opportunities(self, request, queryset):
        updated = queryset.update(is_active=False)
        self.message_user(request, f'{updated} opportunité(s) désactivée(s).')
    deactivate_opportunities.short_description = "Désactiver les opportunités"


@admin.register(Application)
class ApplicationAdmin(admin.ModelAdmin):
    list_display = ('student', 'opportunity', 'status', 'created_at')
    list_filter = ('status', 'created_at', 'opportunity__type')
    search_fields = ('student__email', 'student__first_name', 'student__last_name', 
                    'opportunity__title', 'opportunity__company')
    readonly_fields = ('created_at', 'updated_at')
    actions = ['mark_reviewing', 'mark_shortlisted', 'mark_interviewed', 'mark_accepted', 'mark_rejected']
    
    def mark_reviewing(self, request, queryset):
        updated = queryset.update(status='reviewing')
        self.message_user(request, f'{updated} candidature(s) marquée(s) comme en cours d\'examen.')
    mark_reviewing.short_description = "Marquer comme en cours d'examen"
    
    def mark_shortlisted(self, request, queryset):
        updated = queryset.update(status='shortlisted')
        self.message_user(request, f'{updated} candidature(s) marquée(s) comme présélectionnée(s).')
    mark_shortlisted.short_description = "Marquer comme présélectionnées"
    
    def mark_interviewed(self, request, queryset):
        updated = queryset.update(status='interviewed')
        self.message_user(request, f'{updated} candidature(s) marquée(s) comme interviewée(s).')
    mark_interviewed.short_description = "Marquer comme interviewées"
    
    def mark_accepted(self, request, queryset):
        updated = queryset.update(status='accepted')
        self.message_user(request, f'{updated} candidature(s) marquée(s) comme acceptée(s).')
    mark_accepted.short_description = "Marquer comme acceptées"
    
    def mark_rejected(self, request, queryset):
        updated = queryset.update(status='rejected')
        self.message_user(request, f'{updated} candidature(s) marquée(s) comme rejetée(s).')
    mark_rejected.short_description = "Marquer comme rejetées"