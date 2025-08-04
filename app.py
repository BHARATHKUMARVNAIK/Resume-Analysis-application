
import streamlit as st
import pickle
import re
import docx
import PyPDF2


# loading trained moels
svc_model = pickle.load(open("/Users/bharathkumarvnaik/Downloads/programing/python/AI:Ml-projects/NLP/Resume-scanning-app/data/svc_model.pkl", "rb"))
rf_model = pickle.load(open("/Users/bharathkumarvnaik/Downloads/programing/python/AI:Ml-projects/NLP/Resume-scanning-app/data/rf_model.pkl", "rb"))
clf = pickle.load(open("/Users/bharathkumarvnaik/Downloads/programing/python/AI:Ml-projects/NLP/Resume-scanning-app/data/clf.pkl", "rb"))

tfidf = pickle.load(open("/Users/bharathkumarvnaik/Downloads/programing/python/AI:Ml-projects/NLP/Resume-scanning-app/data/tfidf.pkl", "rb"))
le = pickle.load(open("/Users/bharathkumarvnaik/Downloads/programing/python/AI:Ml-projects/NLP/Resume-scanning-app/data/label_encoder.pkl", "rb"))


# function for cleaning the text in resume
import re
def clean_Resume(txt):
    cleanText = re.sub(r'http\S+', '',txt) # remove URLs
    cleanText = re.sub(r'#\S+', ' ', cleanText) # remove hashtags
    cleanText = re.sub(r'@\S+', ' ', cleanText) # remove mentions
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText) # remove punctuation
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) # remove non-ASCII characters
    cleanText = re.sub(r'\b(RT|cc)\b', ' ', cleanText) # remove RT and cc
    cleanText = re.sub(r'\s+', ' ', cleanText).strip() # remove extra spaces

    return cleanText


# function to extract text from pdf
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# function to extract text from docx
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for para in doc.paragraphs:
        text += para.text + '\n'
    return text


# Function to extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    #  using utf-8 encoding for reading the text file
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        # In case utf-8 fails, try 'latin-1' encoding as a fallback
        text = file.read().decode('latin-1')
    return text


# function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_xtension = uploaded_file.name.split('.')[-1].lower()
    if file_xtension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_xtension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_xtension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a PDF, DOCX, or TXT file.")
    return text



# prediction function
import numpy as np
from scipy.stats import mode

def prediction(input_resume):

    cleaned_text = clean_Resume(input_resume) # cleaning resume text
    vectorized_text = tfidf.transform([cleaned_text]) # vectorizinfg resume text
    vectorized_text = vectorized_text.toarray() # converting to array

    preds=[
        svc_model.predict(vectorized_text),  # SVC prediction
        rf_model.predict(vectorized_text),   # Random Forest prediction
        clf.predict(vectorized_text)          # KNN prediction
    ]
    #predicted_category = clf.predict(vectorized_text) # predicting category
    final_prediction = mode(preds,keepdims=True)[0][0]

    predicted_category_name = le.inverse_transform(final_prediction) # converting to original category

    return predicted_category_name[0]  # returning the predicted category
     

# streamlit app
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")

    st.title("Resume Category Prediction App")
    st.markdown("Upload a resume in PDF, TXT, or DOCX format and get the predicted job category.")

    # File upload section
    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        # Extract text from the uploaded file
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.write("Successfully extracted the text from the uploaded resume.")

            # Display extracted text (optional)
            if st.checkbox("Show extracted text", False):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            # Make prediction
            st.subheader("Predicted Category")
            category = prediction(resume_text)
            st.write(f"The predicted category of the uploaded resume is: **{category}**")

        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")


if __name__ == "__main__":
    main()






