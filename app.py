import streamlit as st
import PyPDF2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Screening System")
st.markdown(
    "Upload a resume and compare it with a job description."
)

#-----------------------------------------
# Load Skills
#-----------------------------------------

with open("skills.txt", "r") as f:
    skills = [line.strip() for line in f.readlines()]

#-----------------------------------------
# Upload Resume
#-----------------------------------------

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste Job Description"
)

#----------------------------------------
# Extract PDF Text
#----------------------------------------

def extract_text(pdf_size):

    pdf_reader = PyPDF2.PdfReader(uploaded_file)

    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

#---------------------------------------
# Main Logic
#---------------------------------------

if uploaded_file and job_description:

    resume_text = extract_text(uploaded_file)

    documents = [
        resume_text,
        job_description
    ]

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )[0][0]

    match_score = round(
        (similarity * 100) + 40,
        2
    )

    if match_score > 100:
        match_score = 100

    st.subheader("🎯 Resume Match Score")

    st.metric(
        "Match %",
        f"{match_score}%"
    )

    st.progress(
        min(int(match_score), 100)
    )

    #----------------------------------------------
    # Skills Detection
    #----------------------------------------------

    resume_skills = []

    for skill in skills:

        if skill.lower() in resume_text.lower():

            resume_skills.append(skill)

    missing_skills = []

    for skill in skills:

        if(
            skill.lower() in job_description.lower() and skill not in resume_skills
        ):
            missing_skills.append(skill)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("✅ Skills Found")

        if resume_skills:

            for skill in resume_skills:
                st.success(skill)

        else:
            st.warning("No skills detected.")

    with col2:

        st.subheader("❌ Missing Skills") 

        if missing_skills:

            for skill in missing_skills:
                st.error(skill)

        else:
            st.success("No missing skills!")

    #-------------------------------------------
    # ATS Score
    #-------------------------------------------

    ats_score = min(
        100,
        match_score + len(resume_skills)
    )

    st.subheader("📈 ATS Compatiblity Score")

    st.metric(
        "ATS Score",
        f"{ats_score:.2f}%"
    )

    #-------------------------------------------
    # Resume Strength
    #-------------------------------------------

    st.subheader("💪 Resume Strength")

    if ats_score < 40:
        st.error("Beginner")

    elif ats_score < 60:
        st.warning("Intermediate")

    elif ats_score < 80:
        st.info("Strong")

    else:
        st.success("Excellent")

    #------------------------------------------
    # Suggestions
    #------------------------------------------                                                               
    
    st.subheader("💡 Suggestions")

    recommendations = []

    if len(missing_skills) > 0:

        recommendations.append(
            "Add projects involving: "
            + ", ".join(missing_skills)
        )

    if "Git" not in resume_skills:

        recommendations.append(
            "Mention Git/Github experience."
        )

    if "AWS" not in resume_skills:

        recommendations.append(
            "Add cloud-related experience."
        )

    if recommendations:

        for rec in recommendations:
            st.info(rec)

    else:

        st.success(
            "Resume is well aligned with the job description."
        )                    