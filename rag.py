"""
RAG Pipeline for Resume Analysis

This module implements the Retrieval-Augmented Generation pipeline using
FAISS vector store and HuggingFace embeddings.
"""

import os
import json
import re
from typing import Dict, List, Optional, Any
from pathlib import Path
import hashlib
import pickle

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI


# Optimized prompt that returns clean JSON
ANALYSIS_SYSTEM_PROMPT = """You are an expert resume reviewer. Analyze the resume against the job description and return ONLY valid JSON.

Output format (strict JSON, no markdown, no explanations):
{
    "missing_skills": ["Python programming - mentioned in JD but not in resume", "SQL databases - required for data analysis role"],
    "improved_points": [
        {"original": "Worked on projects", "improved": "Developed 3 Python applications using Django framework, improving user engagement by 40%", "reason": "Added quantifiable metrics and specific technologies"}
    ],
    "ats_suggestions": ["Include exact keywords from job description", "Use standard section headers like 'Experience' and 'Skills'"],
    "ats_score": 65,
    "matched_keywords": ["JavaScript", "React", "Node.js"],
    "summary": "The candidate shows strong frontend development skills but lacks Python experience required for this role. Consider highlighting transferable skills and pursuing Python training."
}

Guidelines:
- missing_skills: 3-7 specific skills from JD not clearly in resume (with brief reasons)
- improved_points: 3-5 concrete bullet point improvements (original, improved, reason)
- ats_suggestions: 3-5 specific ATS optimization tips for this resume
- ats_score: realistic 0-100 score based on keyword match, formatting, quantifiable achievements
- matched_keywords: 5-10 exact skill matches between resume and JD
- summary: 1-2 sentences assessing fit and key recommendations

CRITICAL: Always provide meaningful, specific content for ALL fields. Never return empty arrays or generic placeholders."""

ANALYSIS_USER_PROMPT = """RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

BEST PRACTICES CONTEXT:
{retrieved_context}

Return ONLY the JSON analysis."""


class RAGPipeline:
    """
    RAG Pipeline for resume analysis.
    Optimized for speed and reliable JSON output.
    """

    def __init__(
        self,
        knowledge_base_dir: str = "knowledge_base",
        embedding_model: str = "all-MiniLM-L6-v2",
        cache_dir: str = ".cache",
        google_api_key: Optional[str] = None,
    ):
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.cache_dir = Path(cache_dir)
        self.embedding_model = embedding_model

        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key

        self._embeddings = None
        self._vector_store = None
        self._llm = None
        self._structured_llm = None
        self._text_splitter = None
        self._kb_loaded = False

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    @property
    def llm(self):
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.1,
                max_output_tokens=2048,
            )
        return self._llm

    @property
    def analysis_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "missing_skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Skills from job description not found in resume"
                },
                "improved_points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "original": {"type": "string"},
                            "improved": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "additionalProperties": True,
                    },
                    "description": "Bullet point improvement suggestions"
                },
                "ats_suggestions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "ATS optimization tips"
                },
                "ats_score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "ATS compatibility score"
                },
                "matched_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Skills present in both resume and job description"
                },
                "summary": {
                    "type": "string",
                    "description": "Brief assessment summary"
                },
            },
            "additionalProperties": True,
        }

    @property
    def structured_llm(self):
        if self._structured_llm is None:
            self._structured_llm = self.llm.with_structured_output(
                self.analysis_schema,
                method="json_schema",
            )
        return self._structured_llm

    @property
    def text_splitter(self):
        if self._text_splitter is None:
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                length_function=len,
            )
        return self._text_splitter

    def _get_cache_key(self) -> str:
        hasher = hashlib.md5()
        if self.knowledge_base_dir.exists():
            for file in sorted(self.knowledge_base_dir.glob("*.md")):
                hasher.update(file.read_bytes())
        return hasher.hexdigest()

    def load_knowledge_base(self, force_rebuild: bool = False) -> None:
        if self._kb_loaded and not force_rebuild:
            return

        cache_key = self._get_cache_key()
        cache_path = self.cache_dir / f"vectorstore_{cache_key}.pkl"

        if not force_rebuild and cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    self._vector_store = pickle.load(f)
                self._kb_loaded = True
                return
            except Exception:
                pass

        documents = []
        if self.knowledge_base_dir.exists():
            for md_file in self.knowledge_base_dir.glob("*.md"):
                content = md_file.read_text()
                documents.append(
                    Document(
                        page_content=content,
                        metadata={"source": md_file.name},
                    )
                )

        if not documents:
            self._kb_loaded = True
            return

        chunks = self.text_splitter.split_documents(documents)
        self._vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings,
        )

        with open(cache_path, "wb") as f:
            pickle.dump(self._vector_store, f)

        self._kb_loaded = True

    def retrieve_context(self, query: str, k: int = 3) -> str:
        if self._vector_store is None:
            self.load_knowledge_base()

        if self._vector_store is None:
            return "Resume best practices: Use action verbs, quantify achievements, keep formatting simple for ATS."

        try:
            docs = self._vector_store.similarity_search(query, k=k)
            context_parts = []
            for doc in docs:
                source = doc.metadata.get("source", "Unknown")
                context_parts.append(f"[{source}]: {doc.page_content[:500]}")
            return "\n".join(context_parts)
        except Exception:
            return "Focus on quantifiable achievements and relevant keywords from the job description."

    def analyze_resume(
        self,
        resume_text: str,
        job_description: str,
    ) -> Dict[str, Any]:
        self.load_knowledge_base()

        retrieval_query = f"ATS tips and resume best practices for: {job_description[:300]}"
        context = self.retrieve_context(retrieval_query, k=3)

        prompt = ANALYSIS_USER_PROMPT.format(
            resume_text=resume_text[:4000],
            job_description=job_description[:2000],
            retrieved_context=context[:1500],
        )

        try:
            # Try structured output first
            try:
                structured_model = self.structured_llm
                response = structured_model.invoke(
                    [
                        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
                )

                if hasattr(response, "dict"):
                    result = response.dict()
                elif isinstance(response, dict):
                    result = response
                else:
                    result = dict(response)

                # Check if we got meaningful content
                if (result.get("missing_skills") or result.get("improved_points") or
                    result.get("ats_suggestions") or result.get("matched_keywords")):
                    return result
                else:
                    raise ValueError("Structured output returned empty content")

            except Exception as structured_error:
                # Fall back to text parsing
                print(f"DEBUG: Structured output failed ({structured_error}), trying text parsing")

                text_model = self.llm
                response = text_model.invoke(
                    [
                        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
                )

                response_text = getattr(response, "text", None) or response.content
                if isinstance(response_text, list):
                    response_text = " ".join(
                        block.get("text", str(block)) if isinstance(block, dict) else str(block)
                        for block in response_text
                    )
                elif not isinstance(response_text, str):
                    response_text = str(response_text)

                result = self._parse_response(response_text)
                return result

        except Exception as e:
            return self._create_fallback_response(str(e))

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        response_text = response_text.strip()

        # Remove markdown code blocks
        if "```json" in response_text:
            response_text = response_text.replace("```json", "").replace("```", "")
        elif response_text.startswith("```"):
            lines = response_text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            response_text = "\n".join(lines)

        # Try direct JSON parse
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Extract JSON from text
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return self._create_fallback_response("Could not parse LLM response")

    def _create_fallback_response(self, error_msg: str) -> Dict[str, Any]:
        return {
            "missing_skills": [
                "Review job description for required technical skills",
                "Check for missing software/tools mentioned in JD",
                "Identify domain-specific knowledge areas",
            ],
            "improved_points": [
                {
                    "original": "Review your bullet points",
                    "improved": "Start with action verb, include metric, show impact",
                    "reason": "Quantified achievements stand out to recruiters",
                }
            ],
            "ats_suggestions": [
                "Use standard section headers (Experience, Education, Skills)",
                "Include exact keywords from job description",
                "Avoid tables, graphics, and complex formatting",
                "Save as PDF with selectable text",
            ],
            "ats_score": 60,
            "matched_keywords": [],
            "summary": (
                "Analysis completed. The AI output could not be parsed cleanly, "
                "but the recommendations above are still valid."
            ),
            "analysis_error": error_msg,
        }

    def quick_analysis(
        self,
        resume_text: str,
        job_description: str,
    ) -> Dict[str, Any]:
        """Fast analysis without RAG context retrieval."""
        prompt = f"""Analyze this resume against the job description. Return ONLY JSON:

{{
    "missing_skills": [],
    "improved_points": [],
    "ats_suggestions": [],
    "ats_score": 0,
    "matched_keywords": [],
    "summary": ""
}}

RESUME: {resume_text[:3000]}
JD: {job_description[:1500]}"""

        try:
            # Try structured output first
            try:
                response = self.structured_llm.invoke([{"role": "user", "content": prompt}])
                if hasattr(response, "dict"):
                    result = response.dict()
                elif isinstance(response, dict):
                    result = response
                else:
                    result = dict(response)

                if result.get("missing_skills") or result.get("improved_points") or result.get("ats_suggestions"):
                    return result
                else:
                    raise ValueError("Empty structured output")

            except Exception as structured_error:
                # Fall back to text parsing
                response = self.llm.invoke([{"role": "user", "content": prompt}])
                response_text = getattr(response, "text", None) or response.content
                if isinstance(response_text, list):
                    response_text = " ".join(
                        block.get("text", str(block)) if isinstance(block, dict) else str(block)
                        for block in response_text
                    )
                return self._parse_response(response_text)

        except Exception as e:
            return self._create_fallback_response(str(e))


def create_pipeline(
    knowledge_base_dir: str = "knowledge_base",
    google_api_key: Optional[str] = None,
) -> RAGPipeline:
    return RAGPipeline(
        knowledge_base_dir=knowledge_base_dir,
        google_api_key=google_api_key,
    )
