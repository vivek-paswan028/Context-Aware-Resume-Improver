"""
LLM Prompt Templates for Resume Analysis

This module contains structured prompts for the Google Gemini API
to analyze resumes against job descriptions using RAG context.
"""

# Main analysis prompt template
ANALYSIS_PROMPT_TEMPLATE = """You are an expert resume reviewer and career coach. Your task is to analyze a resume against a job description and provide actionable improvements.

## Input Data

### Resume Text:
{resume_text}

### Job Description:
{job_description}

### Resume Writing Best Practices (from knowledge base):
{retrieved_context}

## Analysis Requirements

Analyze the resume against the job description considering:

1. **Skill Matching**: Compare skills mentioned in resume vs. required/preferred skills in JD
2. **Experience Alignment**: How well does the candidate's experience match the role requirements?
3. **ATS Optimization**: Is the resume formatted and worded optimally for applicant tracking systems?
4. **Best Practices**: Does the resume follow industry best practices for content and structure?

## Output Format

Return your analysis as a JSON object with the following structure:

```json
{{
    "missing_skills": [
        "skill 1 - briefly explain why it's missing and its importance",
        "skill 2 - briefly explain why it's missing and its importance"
    ],
    "improved_points": [
        {{
            "original": "original bullet point or section",
            "improved": "enhanced version with action verbs and quantifiable metrics",
            "reason": "why this improvement is better"
        }}
    ],
    "ats_suggestions": [
        "specific suggestion 1 for ATS optimization",
        "specific suggestion 2 for ATS optimization"
    ],
    "ats_score": 0-100,
    "matched_keywords": ["keyword1", "keyword2"],
    "summary": "2-3 sentence overall assessment"
}}
```

## Guidelines

- **missing_skills**: List 3-7 specific skills from the JD that are not clearly demonstrated in the resume
- **improved_points**: Provide 3-5 concrete bullet point improvements with before/after comparisons
- **ats_suggestions**: Give 3-5 actionable ATS optimization tips specific to this resume
- **ats_score**: Rate 0-100 based on keyword match, formatting, and ATS best practices
- **matched_keywords**: List 5-10 key skills/terms that ARE present in both resume and JD
- **summary**: Brief overall assessment of candidate fit

Be specific, actionable, and constructive in your feedback. Reference the job description requirements directly.
"""

# Skill extraction prompt
SKILL_EXTRACTION_PROMPT = """Extract all skills mentioned in the following text. Categorize them appropriately.

### Text:
{text}

### Output Format:
Return a JSON object with categorized skills:

```json
{{
    "technical_skills": ["skill1", "skill2"],
    "soft_skills": ["skill1", "skill2"],
    "tools": ["tool1", "tool2"],
    "frameworks": ["framework1", "framework2"],
    "languages": ["language1", "language2"],
    "certifications": ["cert1", "cert2"],
    "domains": ["domain1", "domain2"]
}}
```

Only include skills that are explicitly mentioned. Do not infer or add skills that aren't present.
"""

# ATS score calculation prompt
ATS_SCORE_PROMPT = """Calculate an ATS (Applicant Tracking System) compatibility score for this resume.

### Resume:
{resume_text}

### Job Description:
{job_description}

### Evaluation Criteria:
1. Keyword match rate (0-30 points)
2. Standard section headers present (0-20 points)
3. Quantifiable achievements (0-20 points)
4. Action verbs usage (0-15 points)
5. Relevant skills prominence (0-15 points)

### Output Format:
Return ONLY a JSON object:

```json
{{
    "ats_score": 0-100,
    "keyword_match_percentage": 0-100,
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "critical_missing_keywords": ["keyword1", "keyword2"]
}}
```

Be objective and specific in your scoring.
"""

# Bullet point improvement prompt
BULLET_IMPROVEMENT_PROMPT = """Improve the following resume bullet point using best practices.

### Original Bullet:
{bullet_point}

### Job Description Context:
{job_description}

### Improvement Guidelines:
1. Start with a strong action verb
2. Include quantifiable metrics where possible (use XYZ formula)
3. Highlight impact and results
4. Incorporate relevant keywords from the job description
5. Keep it concise (1-2 lines)

### Output Format:
Return ONLY a JSON object:

```json
{{
    "original": "the original bullet point",
    "improved": "the enhanced version",
    "explanation": "brief explanation of what was improved and why"
}}
```
"""

# Professional summary generator
SUMMARY_GENERATION_PROMPT = """Generate a professional summary for this resume tailored to the job description.

### Resume Experience:
{resume_text}

### Target Job Description:
{job_description}

### Guidelines:
1. Keep it 2-4 lines
2. Highlight most relevant experience for the target role
3. Include years of experience
4. Mention 2-3 key achievements or skills
5. Use strong, confident language

### Output Format:
Return ONLY the professional summary text (no JSON).
"""
