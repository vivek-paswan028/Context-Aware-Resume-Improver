# Context-Aware Resume Improver (RAG)

A production-ready Streamlit web application that uses Retrieval-Augmented Generation (RAG) to analyze resumes against job descriptions and provide targeted improvements.

## 🚀 Features

- **PDF Resume Upload** - Drag-and-drop PDF resume parsing
- **Job Description Analysis** - Paste any job description for comparison
- **RAG-Powered Insights** - Leverages knowledge base of resume best practices and ATS guidelines
- **Missing Skills Detection** - Identifies skills from JD that are missing in your resume
- **Bullet Point Improvements** - AI-powered suggestions for stronger resume bullets
- **ATS Optimization** - Actionable tips to improve Applicant Tracking System compatibility
- **ATS Score (0-100)** - Overall compatibility score with detailed breakdown
- **Keyword Matching** - Visual display of matched vs. missing keywords

## 📁 Project Structure

```
resume-rag/
├── app.py                 # Streamlit frontend application
├── rag.py                 # RAG pipeline (FAISS + embeddings + LLM)
├── utils.py               # PDF parsing, text processing, skill extraction
├── prompts.py             # LLM prompt templates
├── requirements.txt       # Python dependencies
├── knowledge_base/        # Markdown files with best practices
│   ├── resume_best_practices.md
│   ├── ats_optimization.md
│   └── industry_examples.md
├── .env.example           # Environment variable template
└── README.md              # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- Google Gemini API key (get from https://makersuite.google.com/app/apikey)

### Setup Steps

1. **Navigate to the project directory:**
   ```bash
   cd resume-rag
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key:**
   
   Option A - Copy `.env.example` and edit:
   ```bash
   cp .env.example .env
   # Edit .env and add your GOOGLE_API_KEY
   ```

   Option B - Set environment variable directly:
   ```bash
   export GOOGLE_API_KEY="your_api_key_here"
   ```

   Option C - Enter API key in the app sidebar when running

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser:**
   The app will open at `http://localhost:8501`

## 📖 Usage

1. **Enter your Google API key** in the sidebar (if not set via environment)

2. **Upload your resume** (PDF format) using the file uploader

3. **Paste the job description** in the text area

4. **Click "Analyze Resume"** to get:
   - Keyword match percentage
   - Matched vs. missing skills
   - Improved bullet point suggestions
   - ATS optimization tips
   - Overall ATS score (0-100)
   - Resume structure validation

## 🧠 How It Works

### RAG Pipeline

1. **Knowledge Base Loading**: Markdown files containing resume best practices and ATS guidelines are loaded and chunked

2. **Embedding Generation**: Text chunks are embedded using HuggingFace's `all-MiniLM-L6-v2` model

3. **Vector Storage**: Embeddings are stored in FAISS for efficient similarity search

4. **Context Retrieval**: When analyzing a resume, relevant context is retrieved based on the job description

5. **LLM Generation**: Google Gemini analyzes the resume + JD + retrieved context to generate structured feedback

### Skill Extraction

The system uses pattern matching to extract:
- Programming languages
- Frameworks and libraries
- Cloud platforms
- Databases
- Tools and platforms
- Methodologies
- Data science skills
- Soft skills
- Certifications

### ATS Scoring

The ATS score (0-100) is calculated based on:
- Keyword match rate (30 points)
- Standard section headers (20 points)
- Quantifiable achievements (20 points)
- Action verb usage (15 points)
- Relevant skills prominence (15 points)

## 🔧 Configuration

### Changing the Embedding Model

Edit `rag.py` and modify the `embedding_model` parameter:

```python
self.embedding_model = "all-mpnet-base-v2"  # Higher quality, slower
```

### Changing the LLM Model

Edit `rag.py` in the `llm` property:

```python
self._llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",  # More capable, higher cost
    temperature=0.1,
)
```

### Adjusting Retrieval Parameters

In `rag.py`, modify the `analyze_resume` method:

```python
context = self.retrieve_context(retrieval_query, k=5)  # Change k for more/fewer results
```

## 📝 Output Format

The analysis returns structured JSON:

```json
{
    "missing_skills": [
        "Skill - explanation of why it's missing"
    ],
    "improved_points": [
        {
            "original": "Original bullet point",
            "improved": "Enhanced version",
            "reason": "Why this is better"
        }
    ],
    "ats_suggestions": [
        "ATS optimization tip 1",
        "ATS optimization tip 2"
    ],
    "ats_score": 85,
    "matched_keywords": ["skill1", "skill2"],
    "summary": "Overall assessment"
}
```

## 🐛 Troubleshooting

### "No markdown files found in knowledge_base"
Ensure the `knowledge_base` folder exists and contains the `.md` files.

### "Failed to parse PDF"
Try a different PDF file. Some PDFs (especially scanned images) may not have extractable text.

### "API key not valid"
Verify your Google Gemini API key is correct and active.

### "Analysis takes too long"
First run downloads the embedding model (~80MB). Subsequent runs will be faster.

## 📚 Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | Web UI framework |
| langchain | RAG pipeline orchestration |
| langchain-google-genai | Google Gemini integration |
| faiss-cpu | Vector similarity search |
| sentence-transformers | Text embeddings |
| pypdf | PDF text extraction |
| python-dotenv | Environment variable management |

## 🎯 Future Enhancements

- [ ] Support for DOCX file uploads
- [ ] Multiple resume versions comparison
- [ ] Cover letter generation
- [ ] Interview question suggestions based on JD
- [ ] Export analysis as PDF report
- [ ] Integration with LinkedIn profile
- [ ] Industry-specific customization

## 📄 License

MIT License - feel free to use and modify for your needs.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
