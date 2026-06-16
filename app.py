import streamlit as st
import re
import numpy as np

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------
# PAGE SETTINGS
# -------------------------

st.set_page_config(
    page_title="AI CV Matcher Pro",
    layout="centered"
)

st.title("AI CV Matcher Pro")

st.write(
    "AI recruitment analysis using semantic matching + AI capability evaluation"
)

# -------------------------
# OPENAI
# -------------------------

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)

# -------------------------
# LOAD SEMANTIC MODEL
# -------------------------

@st.cache_resource
def load_model():

    return SentenceTransformer(
        "all-mpnet-base-v2"
    )


model = load_model()


# -------------------------
# INPUTS
# -------------------------

cv = st.text_area(
    "Paste CV",
    height=350
)


job = st.text_area(
    "Paste Job Description",
    height=350
)



# -------------------------
# CLEANING
# -------------------------

def clean_text(text):

    if not text:
        return ""

    text = text.lower()

    return re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )



# -------------------------
# SENTENCE SPLIT
# -------------------------

def split_sentences(text):

    sentences = re.split(
        r"[.!?\n]",
        text
    )


    return [

        s.strip()

        for s in sentences

        if len(s.strip()) > 5

    ]



# -------------------------
# SEMANTIC MATCHING
# -------------------------

def semantic_match(cv, job):


    cv_sentences = split_sentences(cv)

    job_sentences = split_sentences(job)



    if not cv_sentences or not job_sentences:

        return 0



    cv_vectors = model.encode(
        cv_sentences
    )


    job_vectors = model.encode(
        job_sentences
    )



    scores = cosine_similarity(
        job_vectors,
        cv_vectors
    )



    best_matches = []


    for row in scores:

        best_matches.append(
            max(row)
        )


    return np.mean(best_matches) * 100


# -------------------------
# AI EVALUATION
# -------------------------

def evaluate_cv(cv, job):

    prompt = f"""

You are a recruitment scoring AI.

Compare this CV with this job description.

Return ONLY this format:


Return ONLY this format:


Eligibility Score: XX/100


Candidate Fit Summary:
(2-3 sentences explaining how this candidate aligns with the role.
Mention strongest relevant experience from the CV.
Consider both qualifications and experience.)


Qualification Alignment:
(1-2 bullet points)
- State whether education/certifications align with the role requirements


Matched Capabilities:
(max 6 workplace capability areas only)

- Capability name only


Evidence Gaps:
(max 3 experience or requirement gaps only)

- Gap only


CV Alignment Feedback:
(2-3 sentences explaining how well the CV matches the role.
Mention strongest evidence and what could improve alignment.)


Rules:
- Qualifications and degrees should be considered when calculating Eligibility Score
- Do not list qualifications as capabilities
- Do not ignore required degrees/certifications
- Separate education from workplace skills
- Focus on evidence from the CV
- Identify transferable skills
- Think like a recruiter reviewing an application


JOB DESCRIPTION:

{job}


CV:

{cv}

"""


    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )


    return response.output_text


    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )


    return response.output_text


# -------------------------
# CV QUALITY GRADER
# -------------------------

def cv_quality(cv):

    score = 0


    words = len(cv.split())


    # Length check
    if 400 <= words <= 900:
        score += 20

    elif words > 250:
        score += 10


    # Achievement language
    action_words = [
        "managed",
        "led",
        "developed",
        "created",
        "implemented",
        "improved",
        "delivered",
        "coordinated",
        "increased",
        "reduced"
    ]


    action_count = sum(
        cv.lower().count(word)
        for word in action_words
    )


    if action_count >= 8:
        score += 25

    elif action_count >= 4:
        score += 15


    # Evidence of impact
    if any(
        char.isdigit()
        for char in cv
    ):
        score += 20


    # Structure
    sections = [
        "experience",
        "education",
        "skills",
        "profile"
    ]


    section_count = sum(
        section in cv.lower()
        for section in sections
    )


    if section_count >= 3:
        score += 20

    else:
        score += 10


    # Professional detail
    if len(cv.split("\n")) > 10:
        score += 15



    if score >= 80:

        grade = "A"

    elif score >= 60:

        grade = "B"

    else:

        grade = "C"


    return grade, score



# -------------------------
# RUN ANALYSIS
# -------------------------

st.info("AI analysis uses API credits. Avoid repeated tests.")


if st.button("Analyse CV"):


    if cv and job:


        if len(cv) > 10000:

            st.error("CV is too long. Please upload a shorter CV.")

            st.stop()


        if len(job) > 10000:

            st.error("Job description is too long. Please shorten it.")

            st.stop()



        cleaned_cv = clean_text(cv)

        cleaned_job = clean_text(job)



        semantic_score = semantic_match(
            cleaned_cv,
            cleaned_job
        )



        ai_analysis = evaluate_cv(
            cv,
            job
        )



        grade, quality_score = cv_quality(cv)



        st.subheader("Match Results")


        st.write(
            f"Semantic Similarity: {semantic_score:.1f}%"
        )



        st.subheader("Capability Analysis")


        st.write(
            ai_analysis
        )



        st.subheader("CV Quality")


        st.write(
            f"Grade: {grade} ({quality_score}/100)"
        )



        if grade == "A":

            st.write(
                "This CV demonstrates strong structure, detail, and professional presentation. "
                "It provides sufficient evidence of experience, skills, and achievements, making it easy for recruiters to identify relevant strengths."
            )


        elif grade == "B":

            st.write(
                "This CV has a solid foundation but could be improved through clearer achievements, stronger action verbs, and more measurable outcomes."
            )


        else:

            st.write(
                "This CV would benefit from stronger structure, clearer evidence of skills, and more detailed achievements."
            )


    else:


        st.error(
            "Please paste both CV and job description"
        )