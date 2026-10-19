import streamlit as st
from transformers import pipeline
from rouge_score import rouge_scorer  

st.set_page_config(page_title="Med-Brief Multi-Model Dashboard", layout="wide")

st.title("Med-Brief Clinical Text Summarizer")
st.write("Compare multiple pretrained transformer models for clinical summarization.")

# Model loading with caching to speed up subsequent runs
@st.cache_resource
def load_model(model_name):
    return pipeline("summarization", model=model_name)

# MODEL SELECTION
model_dict = {
    "T5-small (Balanced)": "t5-small",
    "BART-large-cnn (High Quality)": "facebook/bart-large-cnn",
}

model_choice = st.selectbox("Select Model", list(model_dict.keys()))

summarizer = load_model(model_dict[model_choice])

# INPUT AREA
clinical_text = st.text_area(
    "Enter Clinical Text",
    height=250,
    placeholder="Paste patient discharge summary, clinical notes, or medical report..."
)

# Reference Summary Input
reference_summary = st.text_area(
    "Enter Reference Summary (for ROUGE evaluation)",
    height=150,
    placeholder="Paste ground truth summary here..."
)

# HYPERPARAMETERS
col1, col2, col3 = st.columns(3)

with col1:
    max_len = st.slider("Max Length", 50, 250, 120)

with col2:
    min_len = st.slider("Min Length", 10, 120, 30)

with col3:
    do_sample = st.checkbox("Enable Creativity (Sampling)")

# GENERATE SUMMARY
if st.button("Generate Summary"):
    if not clinical_text.strip():
        st.warning("Please enter clinical text")
    else:
        with st.spinner(f"Generating summary using {model_choice}..."):

            # T5 requires prefix
            if "t5" in model_dict[model_choice]:
                input_text = "summarize: " + clinical_text
            else:
                input_text = clinical_text

            summary = summarizer(
                input_text,
                max_length=max_len,
                min_length=min_len,
                do_sample=do_sample
            )

        generated_summary = summary[0]['summary_text']

        st.subheader("Generated Summary")
        st.success(generated_summary)

        # ROUGE CALCULATION
        if reference_summary.strip():

            scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )

            scores = scorer.score(reference_summary, generated_summary)

            st.subheader("ROUGE Evaluation")

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.metric("ROUGE-1", round(scores['rouge1'].fmeasure, 4))

            with col_b:
                st.metric("ROUGE-2", round(scores['rouge2'].fmeasure, 4))

            with col_c:
                st.metric("ROUGE-L", round(scores['rougeL'].fmeasure, 4))

        else:
            st.info("Enter reference summary above to calculate ROUGE scores.")

# SIDEBAR 
st.sidebar.title("Model Details")
st.sidebar.write("""
**T5-small**  
Balanced speed & quality  

**BART-large-cnn**  
Best summarization quality  
Slow  
""")