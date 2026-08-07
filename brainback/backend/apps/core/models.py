from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.db import models
from django.core.exceptions import ValidationError
from django.conf import settings
from django.core.validators import RegexValidator


class UserManager(BaseUserManager):
    use_in_migrations = True

    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('is_approved', True)
        extra_fields.setdefault('is_active', True)

        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')

        return self.create_user(email, password, **extra_fields)


class User(AbstractUser):
    """
    Custom user model for CPGE platform.
    Handles boursier, supervisor, and admin user types with specific fields.
    """
    # Choices using TextChoices for better readability
    class UserType(models.TextChoices):
        BOURSIER = 'boursier', 'Boursier CPGE'
        SUPERVISOR = 'supervisor', 'Superviseur'
        ADMIN = 'admin', 'Administrateur'

    class BoursierStatus(models.TextChoices):
        ETUDIANT = 'etudiant', 'Étudiant'
        DIPLOME = 'diplome', 'Diplômé'

    class SupervisorType(models.TextChoices):
        MESRSI = 'mesrsi', 'MESRSI'
        CPGE_COORDINATEUR = 'cpge_coordinateur', 'Coordinateur CPGE'
        CPGE_PEDAGOGIQUE = 'cpge_pedagogique', 'Responsable Pédagogique CPGE'
        MENTOR = 'mentor', 'Mentor Alumni'

    # Year validator for promotion and date fields
    year_validator = RegexValidator(r'^\d{4}$', 'Entrez une année au format AAAA.')

    # Basic info
    email = models.EmailField(unique=True, verbose_name="Email")
    first_name = models.CharField(max_length=30, verbose_name="Prénom")
    last_name = models.CharField(max_length=30, verbose_name="Nom")
    user_type = models.CharField(max_length=20, choices=UserType.choices, verbose_name="Type d'utilisateur")

    # Boursier-specific fields
    boursier_status = models.CharField(
        max_length=20,
        choices=BoursierStatus.choices,
        blank=True,
        null=True,
        verbose_name="Statut du boursier"
    )
    promotion_cpge = models.CharField(
        max_length=4,
        blank=True,
        null=True,
        validators=[year_validator],
        verbose_name="Promotion CPGE (année)",
        help_text="Année d'entrée en CPGE (format AAAA)"
    )

    # CPGE establishment choices
    CPGE_ETABLISSEMENT_CHOICES = [
        ("INPHB", "INPHB"),
        ("Dalaba", "Dalaba"),
        ("Lycée du Parc", "Lycée du Parc"),
        ("Autres", "Autres"),
    ]
    cpge_etablissement = models.CharField(
        max_length=20,
        choices=[("", "Sélectionner un établissement")] + CPGE_ETABLISSEMENT_CHOICES,
        blank=True,
        null=True,
        verbose_name="Établissement CPGE"
    )

    # Engineering schools choices
    ENGINEERING_SCHOOLS_CHOICES = [
        ("ESCAE", "ESCAE"),
        ("ESI", "ESI"),
        ("ESMG", "ESMG"),
        ("ESTP", "ESTP"),
        ("ENSEA", "ENSEA"),
        ("ECC", "ECC"),
        ("X", "X"),
        ("ESTP Paris", "ESTP Paris"),
        ("CESI", "CESI"),
        ("ESGT", "ESGT"),
        ("ESTIAM", "ESTIAM"),
        ("EBI", "EBI"),
        ("CNAM", "CNAM"),
        ("EPISEN", "EPISEN"),
        ("ECEMA", "ECEMA"),
        ("EIVP", "EIVP"),
        ("UniLaSalle", "UniLaSalle"),
        ("USMB", "USMB"),
        ("Autres", "Autres"),
    ]
    etablissement_etudes = models.CharField(
        max_length=50,
        choices=[("", "Sélectionner une école")] + ENGINEERING_SCHOOLS_CHOICES,
        blank=True,
        null=True,
        verbose_name="École d'ingénieur (après CPGE)"
    )

    # Mobility and post-engineer schools
    ecoles_mobilite = models.CharField(
        max_length=200,
        blank=True,
        null=True,
        verbose_name="Écoles de mobilité fréquentées"
    )
    ecoles_post_ingenieur = models.CharField(
        max_length=200,
        blank=True,
        null=True,
        verbose_name="Écoles après le cycle ingénieur"
    )

    # Mobility periods
    date_debut_mobilite = models.CharField(
        max_length=4,
        blank=True,
        null=True,
        validators=[year_validator],
        verbose_name="Date de début de la mobilité"
    )
    date_fin_mobilite = models.CharField(
        max_length=4,
        blank=True,
        null=True,
        validators=[year_validator],
        verbose_name="Date de fin de la mobilité"
    )

    # Post-engineer periods
    date_debut_post_ingenieur = models.CharField(
        max_length=4,
        blank=True,
        null=True,
        validators=[year_validator],
        verbose_name="Date de début post-cycle ingénieur"
    )
    date_fin_post_ingenieur = models.CharField(
        max_length=4,
        blank=True,
        null=True,
        validators=[year_validator],
        verbose_name="Date de fin post-cycle ingénieur"
    )

    specialite = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        verbose_name="Spécialité/Formation"
    )

    # CPGE stream choices
    FILIERE_CPGE_CHOICES = [
        ("MP", "MP"),
        ("PC", "PC"),
        ("PSI", "PSI"),
        ("PT", "PT"),
        ("TSI", "TSI"),
        ("BCPST", "BCPST"),
        ("ECS", "ECS"),
        ("Autre", "Autre"),
    ]
    filiere_cpge = models.CharField(
        max_length=10,
        choices=[("", "Sélectionner une filière")] + FILIERE_CPGE_CHOICES,
        blank=True,
        null=True,
        verbose_name="Filière CPGE (2ᵉ année)"
    )

    # Country choices
    COUNTRY_CHOICES = [
        ("CI", "Côte d'Ivoire"),
        ("FR", "France"),
        ("MA", "Maroc"),
        ("CA", "Canada"),
        ("US", "États-Unis"),
        ("GUI", "Guinée"),
        ("DAN", "Danemark"),
        ("Autre", "Autre"),
    ]
    pays_cpge = models.CharField(
        max_length=30,
        choices=[("", "Sélectionner un pays")] + COUNTRY_CHOICES,
        blank=True,
        null=True,
        verbose_name="Pays d'études en CPGE"
    )
    pays_actuel = models.CharField(
        max_length=30,
        choices=[("", "Sélectionner un pays")] + COUNTRY_CHOICES,
        blank=True,
        null=True,
        verbose_name="Pays de résidence actuel"
    )
    pays_cycle_ingenieur = models.CharField(
        max_length=30,
        choices=[("", "Sélectionner un pays")] + COUNTRY_CHOICES,
        blank=True,
        null=True,
        verbose_name="Pays du cycle ingénieur"
    )
    pays_mobilite = models.CharField(
        max_length=30,
        choices=[("", "Sélectionner un pays")] + COUNTRY_CHOICES,
        blank=True,
        null=True,
        verbose_name="Pays de mobilité"
    )
    pays_post_ingenieur = models.CharField(
        max_length=30,
        choices=[("", "Sélectionner un pays")] + COUNTRY_CHOICES,
        blank=True,
        null=True,
        verbose_name="Pays post-cycle ingénieur"
    )

    # Study level choices
    NIVEAU_ETUDES_CHOICES = [
        ("CPGE1", "CPGE 1ᵉ année"),
        ("CPGE2", "CPGE 2ᵉ année"),
        ("ING1", "Cycle ingénieur 1ᵉ année"),
        ("ING2", "Cycle ingénieur 2ᵉ année"),
        ("ING3", "Cycle ingénieur 3ᵉ année"),
        ("LICENCE", "Licence"),
        ("MASTER1", "Master 1"),
        ('BAC+3', 'Bac+3'),
        ('BAC+4', 'Bac+4'),
        ("BAC+5", 'Bac+5'),
        ("MASTER2", "Master 2"),
        ("PHD", "Doctorat"),
        ("Autre", "Autre"),
    ]
    niveau_etudes = models.CharField(
        max_length=10,
        choices=[("", "Sélectionner un niveau")] + NIVEAU_ETUDES_CHOICES,
        blank=True,
        null=True,
        verbose_name="Niveau d'études actuel"
    )
    annee_integration_cycle_ingenieur = models.CharField(
        max_length=4,
        blank=True,
        null=True,
        validators=[year_validator],
        verbose_name="Année d'intégration cycle ingénieur"
    )
    fin_cycle_ingenieur = models.CharField(
        max_length=4,
        blank=True,
        null=True,
        validators=[year_validator],
        verbose_name="Fin du cycle ingénieur"
    )
    date_fin_etudes = models.DateField(
        blank=True,
        null=True,
        verbose_name="Date de fin d'études prévue"
    )

    # Professional info (for graduates)
    entreprise_actuelle = models.CharField(
        max_length=200,
        blank=True,
        null=True,
        verbose_name="Entreprise actuelle"
    )
    poste_actuel = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        verbose_name="Poste actuel"
    )
    secteur_activite = models.CharField(
        max_length=50,
        choices=[
            ("Assurance", "Assurance"),
            ("Bio-industrie", "Bio-industrie"),
            ("Commerce", "Commerce"),
            ("Economie", "Économie"),
            ("Energie", "Énergie"),
            ("Finance-comptabilité", "Finance et comptabilité"),
            ("Genie-civil", "Génie Civil"),
            ("Industrie", "Industrie"),
            ("Informatique", "Informatique"),
            ("Logistique", "Logistique"),
            ("Mines", "Mines"),
            ('Autre', 'Autre'),
        ],
        blank=True,
        null=True,
        verbose_name="Secteur d'activité"
    )

    # Return intentions
    intention_retour = models.CharField(
        max_length=20,
        choices=[
            ('oui', 'Oui, je souhaite retourner'),
            ('peut_etre', 'Peut-être, sous certaines conditions'),
            ('non', 'Non, pas pour le moment'),
            ('deja_retourné', 'Déjà retourné en Guinée'),
        ],
        blank=True,
        null=True,
        verbose_name="Intention de retour"
    )
    conditions_retour = models.TextField(
        blank=True,
        null=True,
        verbose_name="Conditions de retour souhaitées"
    )
    besoins_accompagnement = models.TextField(
        blank=True,
        null=True,
        verbose_name="Besoins d'accompagnement"
    )

    # Priority sector for Guinea
    SECTEUR_PRIORITAIRE_CHOICES = [
        ("education", "Éducation"),
        ("sante", "Santé"),
        ("technologie", "Technologie"),
        ("agriculture", "Agriculture"),
        ("infrastructure", "Infrastructure"),
        ("energie", "Énergie"),
    ]
    secteur_prioritaire = models.CharField(
        max_length=20,
        choices=[("", "Sélectionner un secteur")] + SECTEUR_PRIORITAIRE_CHOICES,
        blank=True,
        null=True,
        verbose_name="Secteur prioritaire pour la Guinée",
    )

    # Support options (checkboxes)
    support_networking = models.BooleanField(
        default=False,
        verbose_name="Réseautage professionnel",
    )
    support_housing = models.BooleanField(
        default=False,
        verbose_name="Aide au logement",
    )
    support_legal = models.BooleanField(
        default=False,
        verbose_name="Assistance administrative/légale",
    )
    support_career = models.BooleanField(
        default=False,
        verbose_name="Orientation professionnelle",
    )

    # Supervisor-specific fields
    supervisor_type = models.CharField(
        max_length=20,
        choices=SupervisorType.choices,
        blank=True,
        null=True,
        verbose_name="Type de superviseur"
    )
    institution_supervision = models.CharField(
        max_length=200,
        blank=True,
        null=True,
        verbose_name="Institution de supervision"
    )

    # Status and approval
    is_approved = models.BooleanField(default=False, verbose_name="Approuvé")
    is_alumni_member = models.BooleanField(default=False, verbose_name="Membre du réseau Alumni")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Date de création")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Dernière modification")

    # Profile photo
    profile_photo = models.ImageField(
        upload_to='profile_photos/',
        blank=True,
        null=True,
        verbose_name="Photo de profil"
    )

    # Social & documents
    linkedin_url = models.URLField(blank=True, null=True, verbose_name="Profil LinkedIn")
    cv_file = models.FileField(upload_to='cvs/', blank=True, null=True, verbose_name="CV (PDF)")

    # Research status
    is_chercheur = models.BooleanField(
        default=False,
        verbose_name="Chercheur / Doctorant",
        help_text="À cocher si le diplômé poursuit en recherche/doctorat"
    )

    # Mentoring availability
    is_mentor = models.BooleanField(default=False, verbose_name="Disponible comme mentor")

    # Removed username field
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['first_name', 'last_name']

    objects = UserManager()

    def __str__(self):
        """String representation for admin and boursier users."""
        if self.user_type == self.UserType.BOURSIER:
            return f"{self.first_name} {self.last_name} - Boursier CPGE {self.promotion_cpge or 'N/A'}"
        return f"{self.first_name} {self.last_name} ({self.email})"

    def clean(self):
        """Custom validation for user creation and update."""
        super().clean()
        # Admins must not have boursier or supervisor status
        if self.user_type == self.UserType.ADMIN and (self.boursier_status or self.supervisor_type):
            raise ValidationError('Les administrateurs ne doivent pas avoir de statut boursier ou superviseur')
        # Only supervisors or graduates can be mentors
        if self.is_mentor and not (
            self.user_type == self.UserType.SUPERVISOR or
            (self.user_type == self.UserType.BOURSIER and self.boursier_status == self.BoursierStatus.DIPLOME)
        ):
            raise ValidationError('Seuls les superviseurs et les diplômés peuvent activer l\'option mentorat')

    def save(self, *args, **kwargs):
        """Custom save logic for user approval and alumni status."""
        self.clean()
        # Admin users are automatically approved and activated
        if self.user_type == self.UserType.ADMIN:
            self.is_approved = True
            self.is_active = True
            self.is_staff = True
            self.is_superuser = True
        else:
            # Alumni status for graduates
            if self.user_type == self.UserType.BOURSIER:
                self.is_alumni_member = self.boursier_status == self.BoursierStatus.DIPLOME
            # Boursiers and supervisors need approval to be active
            if not self.is_approved:
                self.is_active = False
        super().save(*args, **kwargs)

    # Properties for user type checks
    @property
    def is_boursier(self):
        return self.user_type == self.UserType.BOURSIER

    @property
    def is_supervisor(self):
        return self.user_type == self.UserType.SUPERVISOR

    @property
    def is_admin(self):
        return self.user_type == self.UserType.ADMIN

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"

    @property
    def is_etudiant_en_cours(self):
        return self.user_type == self.UserType.BOURSIER and self.boursier_status == self.BoursierStatus.ETUDIANT

    @property
    def is_diplome(self):
        return self.user_type == self.UserType.BOURSIER and self.boursier_status == self.BoursierStatus.DIPLOME

    @property
    def display_promotion(self):
        return self.promotion_cpge or 'N/A'

    class Meta:
        db_table = 'core_user'
        verbose_name = "Utilisateur"
        verbose_name_plural = "Utilisateurs"


class MentorshipRequest(models.Model):
    """
    Model for mentorship requests between students and mentors.
    """
    STATUS = (
        ("PENDING", "En attente"),
        ("ACCEPTED", "Acceptée"),
        ("REJECTED", "Refusée"),
        ("CANCELLED", "Annulée"),
    )

    student = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="mentorship_requests_sent",
        limit_choices_to={"user_type": "boursier"},
        help_text="L'étudiant qui demande un mentorat"
    )
    supervisor = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="mentorship_requests_received",
        limit_choices_to={"is_mentor": True},
        help_text="Le mentor (superviseur ou diplômé)"
    )
    status = models.CharField(max_length=10, choices=STATUS, default="PENDING")
    message = models.TextField(blank=True, help_text="Message de demande")
    created_at = models.DateTimeField(auto_now_add=True)
    responded_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        # Constraints:
        # 1) No duplicate request to the same mentor
        # 2) Only one accepted mentor per student
        constraints = [
            models.UniqueConstraint(fields=["student", "supervisor"], name="unique_request"),
            models.UniqueConstraint(
                fields=["student"],
                condition=models.Q(status="ACCEPTED"),
                name="unique_accepted_mentor_per_student",
            ),
        ]


class Opportunity(models.Model):
    """
    Model for job, internship, thesis, and alternance opportunities.
    """
    TYPE_CHOICES = [
        ('stage', 'Stage'),
        ('emploi', 'Emploi'),
        ('these', 'Thèse'),
        ('alternance', 'Alternance'),
    ]
    EXPERIENCE_LEVEL_CHOICES = [
        ('debutant', 'Débutant'),
        ('junior', 'Junior (1-3 ans)'),
        ('intermediaire', 'Intermédiaire (3-5 ans)'),
        ('senior', 'Senior (5+ ans)'),
    ]
    CONTRACT_TYPE_CHOICES = [
        ('cdi', 'CDI'),
        ('cdd', 'CDD'),
        ('stage', 'Stage'),
        ('these', 'Thèse'),
        ('alternance', 'Alternance'),
    ]

    title = models.CharField(max_length=200, verbose_name="Titre de l'offre")
    company = models.CharField(max_length=100, verbose_name="Entreprise")
    location = models.CharField(max_length=100, verbose_name="Localisation")
    type = models.CharField(
        max_length=20,
        choices=TYPE_CHOICES,
        verbose_name="Type d'opportunité"
    )
    contract_type = models.CharField(
        max_length=20,
        choices=CONTRACT_TYPE_CHOICES,
        verbose_name="Type de contrat"
    )
    experience_level = models.CharField(
        max_length=20,
        choices=EXPERIENCE_LEVEL_CHOICES,
        verbose_name="Niveau d'expérience"
    )
    description = models.TextField(verbose_name="Description du poste")
    requirements = models.TextField(verbose_name="Prérequis et compétences")
    responsibilities = models.TextField(verbose_name="Responsabilités")

    salary_min = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True,
        verbose_name="Salaire minimum"
    )
    salary_max = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True,
        verbose_name="Salaire maximum"
    )
    salary_currency = models.CharField(
        max_length=3,
        default='GNF',
        verbose_name="Devise"
    )

    application_deadline = models.DateField(verbose_name="Date limite de candidature")
    start_date = models.DateField(null=True, blank=True, verbose_name="Date de début souhaitée")

    supervisor = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        limit_choices_to={'user_type': 'supervisor'},
        related_name='posted_opportunities',
        verbose_name="Superviseur"
    )

    is_active = models.BooleanField(default=True, verbose_name="Offre active")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Date de création")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Dernière modification")

    class Meta:
        verbose_name = "Opportunité"
        verbose_name_plural = "Opportunités"
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.title} - {self.company}"


class Application(models.Model):
    """
    Model for student applications to opportunities.
    """
    STATUS_CHOICES = [
        ('pending', 'En attente'),
        ('reviewing', 'En cours d\'examen'),
        ('shortlisted', 'Présélectionné'),
        ('interviewed', 'Entretien effectué'),
        ('accepted', 'Accepté'),
        ('rejected', 'Refusé'),
        ('withdrawn', 'Retiré'),
    ]

    opportunity = models.ForeignKey(
        Opportunity,
        on_delete=models.CASCADE,
        related_name='applications',
        verbose_name="Opportunité"
    )
    student = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        limit_choices_to={'user_type': 'boursier'},
        related_name='applications',
        verbose_name="Étudiant"
    )

    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending',
        verbose_name="Statut"
    )

    cover_letter = models.TextField(blank=True, null=True, verbose_name="Lettre de motivation")
    cv_file = models.FileField(
        upload_to='applications/cvs/',
        verbose_name="CV (PDF)",
        help_text="Format PDF uniquement"
    )
    additional_documents = models.FileField(
        upload_to='applications/documents/',
        null=True,
        blank=True,
        verbose_name="Documents supplémentaires"
    )

    supervisor_notes = models.TextField(
        blank=True,
        verbose_name="Notes du superviseur",
        help_text="Notes internes sur la candidature"
    )

    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Date de candidature")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Dernière modification")

    class Meta:
        verbose_name = "Candidature"
        verbose_name_plural = "Candidatures"
        ordering = ['-created_at']
        # Constraint: only one application per student per opportunity
        constraints = [
            models.UniqueConstraint(
                fields=['opportunity', 'student'],
                name='unique_application'
            )
        ]

    def __str__(self):
        return f"Candidature de {self.student.full_name} pour {self.opportunity.title}"