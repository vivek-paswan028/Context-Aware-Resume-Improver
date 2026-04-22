"""
Context-Aware Resume Improver - Streamlit Application

A RAG-powered web app that analyzes resumes against job descriptions
and provides targeted improvements using Google Gemini AI.
"""

import os
import json
from typing import Optional, Dict, Any, List

import streamlit as st
from dotenv import load_dotenv

from rag import create_pipeline, RAGPipeline
from utils import (
    pdf_to_text,
    extract_skills,
    get_all_skills,
    calculate_keyword_match,
    validate_resume_structure,
    count_action_verbs,
    extract_bullet_points,
)

# Load environment variables
load_dotenv()

# Set environment variables to suppress warnings
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Page configuration
st.set_page_config(
    page_title="Resume Improver (RAG)",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .result-section {
        background-color: #f9fafb;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .skill-tag {
        display: inline-block;
        background-color: #e5e7eb;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        margin: 0.25rem;
        font-size: 0.875rem;
    }
    .missing-skill {
        background-color: #fecaca;
    }
    .matched-skill {
        background-color: #bbf7d0;
    }
    .ats-score-high {
        color: #059669;
    }
    .ats-score-medium {
        color: #d97706;
    }
    .ats-score-low {
        color: #dc2626;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    """Initialize session state variables."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "resume_text" not in st.session_state:
        st.session_state.resume_text = None
    if "resume_skills" not in st.session_state:
        st.session_state.resume_skills = None
    if "jd_text" not in st.session_state:
        st.session_state.jd_text = None


def check_api_key() -> bool:
    """Check if API key is loaded from environment."""
    return os.getenv("GOOGLE_API_KEY") is not None


def setup_sidebar():
    """Setup sidebar with info and API key status."""
    with st.sidebar:
        st.header("⚙️ Settings")

        # Check API key status
        api_key_loaded = check_api_key()

        if api_key_loaded:
            st.success("✅ API key loaded from environment")
        else:
            st.error("❌ API key not found")
            st.markdown(
                """
                **To set up your API key:**
                1. Copy `.env.example` to `.env`
                2. Add your API key from https://makersuite.google.com/app/apikey
                """
            )

        st.divider()

        st.subheader("ℹ️ How it works")
        st.markdown(
            """
        1. **Upload** your resume (PDF format)
        2. **Paste** the job description
        3. **Click** Analyze to get:
           - Missing skills identification
           - Improved bullet point suggestions
           - ATS optimization tips
           - Overall ATS compatibility score
        """
        )

        st.divider()

        st.subheader("📚 Knowledge Base")
        st.markdown(
            """
        The RAG system uses:
        - Resume writing best practices
        - ATS optimization guidelines
        - Industry-standard examples
        """
        )

        st.divider()

        st.subheader("🔒 Security")
        st.info(
            "API key is loaded from `.env` file and never exposed in the UI."
        )

        if st.button("🗑️ Clear Analysis", use_container_width=True):
            st.session_state.analysis_result = None
            st.session_state.resume_text = None
            st.session_state.resume_skills = None
            st.session_state.jd_text = None
            st.rerun()

        return api_key_loaded


def render_header():
    """Render the main header."""
    st.markdown('<h1 class="main-header">📄 Context-Aware Resume Improver</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Get AI-powered feedback to tailor your resume for any job description</p>',
        unsafe_allow_html=True,
    )


def render_resume_upload() -> Optional[str]:
    """Render resume upload section."""
    st.subheader("1️⃣ Upload Your Resume")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload your resume in PDF format",
        key="resume_uploader",
    )

    if uploaded_file is not None:
        try:
            with st.spinner("📖 Extracting text from PDF..."):
                resume_text = pdf_to_text(uploaded_file)

            if not resume_text.strip():
                st.error("Could not extract text from the PDF. Please try another file.")
                return None

            st.success(f"✅ Extracted {len(resume_text)} characters")

            with st.expander("👁️ Preview Extracted Text"):
                st.text(resume_text[:2000] + ("..." if len(resume_text) > 2000 else ""))

            return resume_text

        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return None

    return None


def render_job_description_input() -> Optional[str]:
    """Render job description input section."""
    st.subheader("2️⃣ Enter Job Description")

    jd_input = st.text_area(
        "Paste the job description here",
        height=250,
        placeholder="Paste the full job description including requirements, responsibilities, and preferred qualifications...",
        key="jd_input",
        value=st.session_state.get("jd_text", ""),
    )

    if jd_input and len(jd_input.strip()) < 50:
        st.warning("Please enter a more detailed job description for better analysis.")
        return None

    return jd_input.strip() if jd_input else None


def render_skill_analysis(analysis_result: Dict[str, Any]):
    """Render the skill analysis section."""
    st.subheader("🎯 Skill Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ✅ Matched Skills")
        matched = analysis_result.get("matched_keywords", [])
        if matched:
            for skill in matched[:15]:
                st.markdown(f'<span class="skill-tag matched-skill">{skill}</span>', unsafe_allow_html=True)
            if len(matched) > 15:
                st.caption(f"+ {len(matched) - 15} more")
        else:
            st.info("No matched skills identified.")

    with col2:
        st.markdown("#### ❌ Missing Skills")
        missing = analysis_result.get("missing_skills", [])
        if missing:
            for skill_info in missing[:10]:
                if isinstance(skill_info, dict):
                    st.markdown(f"**{skill_info.get('skill', skill_info)}**")
                    if "reason" in skill_info:
                        st.caption(skill_info["reason"])
                else:
                    st.markdown(f"• {skill_info}")
            if len(missing) > 10:
                st.caption(f"+ {len(missing) - 10} more")
        else:
            st.info("No missing skills identified - great job!")


def generate_fallback_ats_suggestions(resume_text: str, ats_score: int) -> List[str]:
    """Generate fallback ATS suggestions based on resume analysis."""
    suggestions = []

    # Analyze resume structure
    structure = validate_resume_structure(resume_text)

    # Basic ATS suggestions
    if not structure.get("has_experience", False):
        suggestions.append("Add a clear 'Experience' or 'Work Experience' section header")

    if not structure.get("has_education", False):
        suggestions.append("Include an 'Education' section with your degrees and institutions")

    if not structure.get("has_skills", False):
        suggestions.append("Create a dedicated 'Skills' section with relevant technical skills")

    if not structure.get("has_contact_info", False):
        suggestions.append("Ensure contact information is prominently displayed at the top")

    if not structure.get("has_bullet_points", False):
        suggestions.append("Use bullet points for job descriptions instead of paragraphs")

    if not structure.get("has_metrics", False):
        suggestions.append("Include quantifiable achievements and metrics in your experience descriptions")

    # Score-based suggestions
    if ats_score < 60:
        suggestions.extend([
            "Use exact keywords from the job description throughout your resume",
            "Save your resume as a PDF to preserve formatting",
            "Avoid tables, graphics, or complex formatting that ATS systems can't read",
            "Use standard fonts (Arial, Calibri, Times New Roman) in 10-12pt size",
            "Keep file name simple (e.g., 'John_Doe_Resume.pdf')",
        ])
    elif ats_score < 80:
        suggestions.extend([
            "Review job description for industry-specific keywords to include",
            "Ensure consistent date formatting (MM/YYYY)",
            "Use action verbs at the start of bullet points (Managed, Developed, Created)",
            "Limit resume to 1-2 pages for most positions",
        ])

    # Always include these best practices
    if len(suggestions) < 5:
        suggestions.extend([
            "Tailor your resume for each job application with relevant keywords",
            "Proofread carefully for spelling and grammar errors",
            "Use a clean, professional format with consistent spacing",
        ])

    return suggestions[:8]  # Limit to 8 suggestions


def render_improved_points(analysis_result: Dict[str, Any]):
    """Render the improved bullet points section."""
    st.subheader("✨ Improved Bullet Points")

    improved_points = analysis_result.get("improved_points", [])

    if not improved_points:
        st.info("💡 **Tips for Better Bullet Points:**")
        st.markdown("""
        - **Start with action verbs**: Use words like "Developed", "Managed", "Created", "Improved"
        - **Include metrics**: Add numbers to show impact (e.g., "Increased sales by 25%")
        - **Be specific**: Mention tools, technologies, and methodologies used
        - **Quantify achievements**: Use percentages, dollar amounts, or other measurable results
        - **Keep it concise**: Aim for 1-2 lines per bullet point
        """)
        return

    for i, point in enumerate(improved_points, 1):
        with st.expander(f"💡 Improvement {i}", expanded=False):
            if isinstance(point, dict):
                original = point.get("original", "N/A")
                improved = point.get("improved", "N/A")
                reason = point.get("reason", "")

                st.markdown("**📝 Original:**")
                st.write(original)

                st.markdown("**✨ Improved:**")
                st.write(improved)

                if reason:
                    st.caption(f"💬 **Why this improves it:** {reason}")
            else:
                st.write(point)


def render_ats_suggestions(analysis_result: Dict[str, Any], resume_text: str):
    """Render ATS suggestions section."""
    st.subheader("📈 ATS Optimization")

    ats_score = analysis_result.get("ats_score", 0)

    col1, col2 = st.columns([1, 2])

    with col1:
        if ats_score >= 80:
            emoji = "🟢"
            label = "Excellent"
        elif ats_score >= 60:
            emoji = "🟡"
            label = "Good"
        else:
            emoji = "🔴"
            label = "Needs Improvement"

        st.markdown(f"### {emoji} ATS Score")
        st.markdown(f"# {ats_score}/100")
        st.markdown(f"*{label}*")

    with col2:
        structure = validate_resume_structure(resume_text)
        st.markdown("#### Resume Structure Check")

        checks = [
            ("Has Experience Section", structure.get("has_experience", False)),
            ("Has Education Section", structure.get("has_education", False)),
            ("Has Skills Section", structure.get("has_skills", False)),
            ("Has Contact Info", structure.get("has_contact_info", False)),
            ("Has Bullet Points", structure.get("has_bullet_points", False)),
            ("Has Quantifiable Metrics", structure.get("has_metrics", False)),
        ]

        for check_name, passed in checks:
            icon = "✅" if passed else "❌"
            st.write(f"{icon} {check_name}")

    st.divider()
    st.markdown("#### 🔧 ATS Suggestions")

    suggestions = analysis_result.get("ats_suggestions", [])

    if not suggestions:
        # Generate fallback suggestions based on resume analysis
        ats_score = analysis_result.get("ats_score", 50)
        suggestions = generate_fallback_ats_suggestions(resume_text, ats_score)

    if suggestions:
        for i, suggestion in enumerate(suggestions, 1):
            if isinstance(suggestion, dict):
                st.markdown(f"**{i}.** {suggestion.get('suggestion', suggestion.get('title', ''))}")
                if "details" in suggestion:
                    st.caption(suggestion["details"])
            else:
                st.markdown(f"**{i}.** {suggestion}")
    else:
        st.info("No specific ATS suggestions. Your resume appears ATS-friendly!")


def render_summary(analysis_result: Dict[str, Any]):
    """Render the overall summary section."""
    st.subheader("📋 Overall Assessment")

    summary = analysis_result.get("summary", "")
    analysis_error = analysis_result.get("analysis_error", "")

    if summary and summary != "Analysis completed.":
        st.info(summary)
    else:
        # Show detailed breakdown if no custom summary
        col1, col2 = st.columns(2)

        with col1:
            ats_score = analysis_result.get("ats_score", 50)
            st.metric("ATS Score", f"{ats_score}/100")

            matched = analysis_result.get("matched_keywords", [])
            missing = analysis_result.get("missing_skills", [])
            st.metric("Matched Skills", len(matched))
            st.metric("Missing Skills", len(missing))

        with col2:
            suggestions = analysis_result.get("ats_suggestions", [])
            improvements = analysis_result.get("improved_points", [])
            st.metric("ATS Suggestions", len(suggestions))
            st.metric("Bullet Point Improvements", len(improvements))

        st.info("Analysis completed. Review the detailed sections above for specific recommendations.")

    if analysis_error:
        with st.expander("⚠️ AI parsing details"):
            st.write(analysis_error)


def validate_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure result has all required fields with proper types."""
    defaults = {
        "missing_skills": [],
        "improved_points": [],
        "ats_suggestions": [],
        "ats_score": 50,
        "matched_keywords": [],
        "summary": "Analysis completed.",
        "analysis_error": "",
        "keyword_match_percentage": 0,
    }

    for key, default_value in defaults.items():
        if key not in result:
            result[key] = default_value
        elif result[key] is None:
            result[key] = default_value

    # Ensure ats_score is a number
    if not isinstance(result.get("ats_score"), (int, float)):
        try:
            result["ats_score"] = int(result["ats_score"])
        except (ValueError, TypeError):
            result["ats_score"] = 50

    # Ensure lists are lists
    for list_field in ["missing_skills", "improved_points", "ats_suggestions", "matched_keywords"]:
        if not isinstance(result.get(list_field), list):
            result[list_field] = []

    return result


def main():
    """Main application function."""
    init_session_state()
    api_key_loaded = setup_sidebar()
    render_header()

    if not api_key_loaded:
        st.warning("⚠️ API key not configured. Please set up your `.env` file to continue.")
        st.stop()

    col1, col2 = st.columns([2, 1])

    with col1:
        resume_text = render_resume_upload()
    with col2:
        jd_text = render_job_description_input()

    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🚀 Analyze Resume",
            type="primary",
            use_container_width=True,
            disabled=(not resume_text or not jd_text),
        )

    if analyze_button and resume_text and jd_text:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("⏳ Loading knowledge base...")
            progress_bar.progress(10)

            if st.session_state.pipeline is None:
                st.session_state.pipeline = create_pipeline(
                    knowledge_base_dir="knowledge_base",
                )

            status_text.text("📊 Extracting skills...")
            progress_bar.progress(20)

            resume_skills = extract_skills(resume_text)
            jd_skills = extract_skills(jd_text)

            st.session_state.resume_text = resume_text
            st.session_state.resume_skills = resume_skills
            st.session_state.jd_text = jd_text

            status_text.text("🤖 Analyzing with AI... (this may take 15-30 seconds)")
            progress_bar.progress(40)

            result = st.session_state.pipeline.analyze_resume(
                resume_text=resume_text,
                job_description=jd_text,
            )

            status_text.text("🔗 Calculating keyword match...")
            progress_bar.progress(80)

            all_resume_skills = get_all_skills(resume_skills)
            all_jd_skills = get_all_skills(jd_skills)

            match_pct, matched, missing = calculate_keyword_match(all_resume_skills, all_jd_skills)

            result["matched_keywords"] = list(matched)[:15]
            result["keyword_match_percentage"] = round(match_pct, 1)

            result = validate_result(result)

            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")

            st.session_state.analysis_result = result
            st.success("✅ Analysis complete!")

        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.caption("Make sure you have a valid Google API key.")
            progress_bar.empty()
            status_text.empty()

    if st.session_state.analysis_result:
        st.divider()

        result = st.session_state.analysis_result

        match_pct = result.get("keyword_match_percentage", 0)
        st.markdown(f"#### Keyword Match: {match_pct}%")
        st.progress(min(match_pct / 100, 1.0))

        render_skill_analysis(result)
        render_improved_points(result)
        render_ats_suggestions(result, st.session_state.resume_text)
        render_summary(result)

        with st.expander("🔍 Debug: Raw Analysis Result"):
            st.json(result)


if __name__ == "__main__":
    main()
