import streamlit as st
import os
import json
import pandas as pd
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.callbacks import get_openai_callback
import base64

load_dotenv()

KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key=KEY, model_name="gpt-3.5-turbo", temperature=0.5)

RESPONSE_JSON = {
    "1": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    # Include other questions as per your requirement
}

TEMPLATE = """
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{RESPONSE_JSON}
"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "RESPONSE_JSON"],
    template=TEMPLATE
)

TEMPLATE2 = """
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at par with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}
Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt = PromptTemplate(
    input_variables=["subject", "quiz"],
    template=TEMPLATE2
)

quiz_chain = LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)
review_chain = LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)

generate_evaluate_chain = SequentialChain(chains=[quiz_chain, review_chain],
                                          input_variables=["text", "number", "subject", "tone", "RESPONSE_JSON"],
                                          output_variables=["quiz", "review"], verbose=True)

def generate_mcqs(text, number, subject, tone):
    with get_openai_callback() as cb:
        response = generate_evaluate_chain({
            "text": text,
            "number": number,
            "subject": subject,
            "tone": tone,
            "RESPONSE_JSON": json.dumps(RESPONSE_JSON)
        })

    quiz = response.get("quiz")
    quiz = json.loads(quiz)

    quiz_table_data = []
    for key, value in quiz.items():
        mcq = value["mcq"]
        options = " | ".join([f"{option}: {option_value}" for option, option_value in value["options"].items()])
        correct = value["correct"]
        quiz_table_data.append({"MCQ": mcq, "Choices": options, "Correct": correct})

    quiz_df = pd.DataFrame(quiz_table_data)
    return quiz_df

def main():
    st.title("MCQs Creator Application with LangChain :parrot::chains:")

    uploaded_file = st.file_uploader("Upload a file", type=['txt'])

    if uploaded_file:
        file_contents = uploaded_file.getvalue()
        st.text("File uploaded successfully!")

        number = st.number_input('Select number of MCQs', min_value=1, max_value=20, value=5)
        subject = st.text_input('Enter subject')
        tone = st.selectbox('Select question complexity level',
                            ['Basic Level', 'Intermediate Level', 'Advanced Level'])

        if st.button('Generate MCQs'):
            generated_mcqs = generate_mcqs(file_contents, number, subject.lower(), tone.lower())
            st.write(generated_mcqs)
            st.markdown(get_csv_download_link(generated_mcqs), unsafe_allow_html=True)

def get_csv_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="generated_mcqs.csv">Download CSV File</a>'
    return href


if __name__ == "__main__":
    main()
