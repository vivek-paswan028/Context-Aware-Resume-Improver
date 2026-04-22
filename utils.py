"""
Utility Functions for Resume Processing

This module provides PDF parsing, text cleaning, and skill extraction utilities
for the Resume Improver application.
"""

import re
from typing import Dict, List, Set, Tuple
from pypdf import PdfReader
from io import BytesIO


# Comprehensive skill patterns for extraction
SKILL_PATTERNS = {
    "programming_languages": [
        r"\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|Go|Rust|Ruby|PHP|Swift|Kotlin|Scala|R|SQL|HTML|CSS|Shell|Bash|Perl|MATLAB|SAS|Julia)\b",
    ],
    "frameworks_libraries": [
        r"\b(React|Angular|Vue\.?js|Svelte|Django|Flask|FastAPI|Spring|Express\.?js|Next\.?js|Nuxt|TensorFlow|PyTorch|scikit-learn|Pandas|NumPy|Keras|Redux|MobX|Tailwind|Bootstrap|jQuery|Laravel|Rails|ASP\.NET)\b",
    ],
    "cloud_platforms": [
        r"\b(AWS|Azure|GCP|Google Cloud|Amazon Web Services|EC2|S3|Lambda|CloudFormation|Terraform|Kubernetes|Docker|OpenShift|Heroku|Vercel|Netlify)\b",
    ],
    "databases": [
        r"\b(PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch|Cassandra|DynamoDB|SQLite|Oracle|SQL Server|BigQuery|Snowflake|MariaDB|CouchDB|Neo4j)\b",
    ],
    "tools_platforms": [
        r"\b(Git|GitHub|GitLab|Bitbucket|Jenkins|CircleCI|Travis|GitLab CI|Jira|Confluence|Trello|Asana|Figma|Sketch|Adobe|Postman|Insomnia|Wireshark|Splunk|Datadog|New Relic|Grafana|Prometheus)\b",
    ],
    "methodologies": [
        r"\b(Agile|Scrum|Kanban|DevOps|CI/?CD|TDD|BDD|Microservices|REST|GraphQL|gRPC|SOAP|OAuth|JWT|Linux|Unix|Windows Server|VMware|VirtualBox)\b",
    ],
    "data_science": [
        r"\b(Machine Learning|Deep Learning|NLP|Computer Vision|Data Analysis|Data Visualization|Statistical Analysis|A/?B Testing|Predictive Modeling|Time Series|Feature Engineering|Hyperparameter Tuning|XGBoost|LightGBM|CatBoost|Spark|Hadoop|Airflow|Kafka)\b",
    ],
    "soft_skills": [
        r"\b(Leadership|Communication|Teamwork|Problem Solving|Critical Thinking|Time Management|Adaptability|Creativity|Emotional Intelligence|Conflict Resolution|Mentoring|Public Speaking|Negotiation|Strategic Planning|Project Management)\b",
    ],
    "certifications": [
        r"\b(AWS Certified|Google Cloud Certified|Azure Certified|PMP|CSM|CISSP|CompTIA|CCNA|CCNP|RHCE|VCP|MCSE|CFA|CPA|PE|Six Sigma|ITIL|CISM|CEH|OSCP)\b",
    ],
}

# Common action verbs for resume analysis
ACTION_VERBS = {
    "leadership": [
        "Led", "Directed", "Orchestrated", "Spearheaded", "Championed",
        "Mentored", "Supervised", "Managed", "Coordinated", "Facilitated"
    ],
    "technical": [
        "Developed", "Engineered", "Architected", "Implemented", "Optimized",
        "Automated", "Integrated", "Designed", "Built", "Created", "Programmed"
    ],
    "analysis": [
        "Analyzed", "Evaluated", "Assessed", "Researched", "Investigated",
        "Diagnosed", "Audited", "Reviewed", "Examined", "Tested"
    ],
    "achievement": [
        "Achieved", "Accomplished", "Attained", "Exceeded", "Surpassed",
        "Delivered", "Generated", "Increased", "Reduced", "Improved"
    ],
    "communication": [
        "Communicated", "Presented", "Negotiated", "Collaborated", "Authored",
        "Published", "Documented", "Trained", "Advised", "Persuaded"
    ],
}


def pdf_to_text(pdf_file) -> str:
    """
    Extract text from an uploaded PDF file.

    Args:
        pdf_file: File-like object from Streamlit uploader

    Returns:
        Extracted text content as string
    """
    try:
        # Read the PDF file
        pdf_reader = PdfReader(BytesIO(pdf_file.read()))
        text = ""

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        return clean_text(text)

    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {str(e)}")


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text.

    Args:
        text: Raw text content

    Returns:
        Cleaned text with consistent formatting
    """
    if not text:
        return ""

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters that might be artifacts
    text = re.sub(r'[^\w\s.,;:!?()\-/+$%&@#]', ' ', text)

    # Normalize line breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)

    # Remove leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


def extract_skills(text: str) -> Dict[str, List[str]]:
    """
    Extract skills from text using pattern matching.

    Args:
        text: Text content to analyze

    Returns:
        Dictionary of categorized skills
    """
    if not text:
        return {category: [] for category in SKILL_PATTERNS.keys()}

    text_lower = text.lower()
    extracted_skills = {category: set() for category in SKILL_PATTERNS.keys()}

    for category, patterns in SKILL_PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Normalize the match
                normalized = match.strip()
                if normalized and len(normalized) > 1:
                    extracted_skills[category].add(normalized)

    # Convert sets to sorted lists
    return {category: sorted(list(skills)) for category, skills in extracted_skills.items()}


def get_all_skills(extracted: Dict[str, List[str]]) -> Set[str]:
    """
    Get a flat set of all extracted skills.

    Args:
        extracted: Dictionary of categorized skills

    Returns:
        Flat set of all skills
    """
    all_skills = set()
    for skills in extracted.values():
        all_skills.update(skills)
    return all_skills


def count_action_verbs(text: str) -> Dict[str, int]:
    """
    Count action verbs used in the resume.

    Args:
        text: Resume text

    Returns:
        Dictionary with counts per category
    """
    counts = {category: 0 for category in ACTION_VERBS.keys()}
    text_lower = text.lower()

    for category, verbs in ACTION_VERBS.items():
        for verb in verbs:
            # Count occurrences (word boundary check)
            pattern = r'\b' + re.escape(verb.lower()) + r'\b'
            counts[category] += len(re.findall(pattern, text_lower))

    return counts


def extract_bullet_points(text: str) -> List[str]:
    """
    Extract bullet points from resume text.

    Args:
        text: Resume text

    Returns:
        List of bullet point strings
    """
    # Match common bullet point patterns
    bullet_patterns = [
        r'^[\s]*[-•●○▪▸►]+\s*(.+)$',
        r'^[\s]*\*\s*(.+)$',
        r'^[\s]*\d+[\.)]\s*(.+)$',
    ]

    bullets = []
    for line in text.split('\n'):
        for pattern in bullet_patterns:
            match = re.match(pattern, line, re.MULTILINE)
            if match:
                bullet_text = match.group(1).strip()
                if len(bullet_text) > 10:  # Filter out very short lines
                    bullets.append(bullet_text)
                break

    return bullets


def calculate_keyword_match(resume_skills: Set[str], jd_skills: Set[str]) -> Tuple[float, Set[str], Set[str]]:
    """
    Calculate keyword match between resume and job description.

    Args:
        resume_skills: Skills extracted from resume
        jd_skills: Skills extracted from job description

    Returns:
        Tuple of (match_percentage, matched_skills, missing_skills)
    """
    if not jd_skills:
        return 100.0, set(), set()

    # Normalize for comparison
    resume_lower = {s.lower() for s in resume_skills}
    jd_lower = {s.lower() for s in jd_skills}

    matched = resume_lower.intersection(jd_lower)
    missing = jd_lower - resume_lower

    match_percentage = (len(matched) / len(jd_lower)) * 100 if jd_lower else 100.0

    return match_percentage, matched, missing


def extract_sections(text: str) -> Dict[str, str]:
    """
    Extract standard resume sections from text.

    Args:
        text: Resume text

    Returns:
        Dictionary mapping section names to content
    """
    sections = {}

    # Common section headers
    section_patterns = [
        (r'(?:^|\n)\s*(?:PROFESSIONAL|WORK)?\s*EXPERIENCE\s*(?:$|\n)', 'experience'),
        (r'(?:^|\n)\s*EDUCATION\s*(?:$|\n)', 'education'),
        (r'(?:^|\n)\s*(?:TECHNICAL\s*)?SKILLS?\s*(?:$|\n)', 'skills'),
        (r'(?:^|\n)\s*(?:PROFESSIONAL\s*)?SUMMARY\s*(?:$|\n)', 'summary'),
        (r'(?:^|\n)\s*(?:PROJECTS|PERSONAL\s*PROJECTS)\s*(?:$|\n)', 'projects'),
        (r'(?:^|\n)\s*CERTIFICATIONS?\s*(?:$|\n)', 'certifications'),
    ]

    text_lower = text.lower()

    for pattern, section_name in section_patterns:
        match = re.search(pattern, text_lower)
        if match:
            start = match.end()
            # Find next section
            next_section = len(text)
            for other_pattern, _ in section_patterns:
                other_match = re.search(other_pattern, text_lower[start:])
                if other_match and other_match.start() < next_section:
                    next_section = start + other_match.start()

            sections[section_name] = text[start:next_section].strip()

    return sections


def has_quantifiable_metrics(text: str) -> bool:
    """
    Check if text contains quantifiable metrics.

    Args:
        text: Text to analyze

    Returns:
        True if metrics are found
    """
    metric_patterns = [
        r'\d+%',                    # Percentages
        r'\$\d+[KMBkmb]?',          # Dollar amounts
        r'\d+\s*(?:users|customers|clients)',  # User counts
        r'\d+\s*(?:lines?|files?|records?)',   # Volume metrics
        r'\d+\s*(?:ms|seconds?|minutes?|hours?|days?|weeks?|months?)',  # Time metrics
        r'\d+x\s*(?:faster|slower|improvement|reduction)',  # Multipliers
        r'from\s*\d+\s*to\s*\d+',   # Range improvements
    ]

    for pattern in metric_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def validate_resume_structure(text: str) -> Dict[str, bool]:
    """
    Validate that resume has essential sections.

    Args:
        text: Resume text

    Returns:
        Dictionary of section presence checks
    """
    sections = extract_sections(text)

    return {
        'has_experience': 'experience' in sections,
        'has_education': 'education' in sections,
        'has_skills': 'skills' in sections,
        'has_summary': 'summary' in sections,
        'has_contact_info': bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)),
        'has_bullet_points': len(extract_bullet_points(text)) > 0,
        'has_metrics': has_quantifiable_metrics(text),
    }
