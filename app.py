import os
import time
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback
from langchain_community.cache import InMemoryCache
import langchain
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Configure LangChain caching
langchain.llm_cache = InMemoryCache()

# Global variable to store the VectorStore
VectorStore = None

def create_and_save_vectorstore(file_path, save_path):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    texts = []
    for chunk in pd.read_csv(file_path, chunksize=500):
        text = "\n".join(chunk.values.astype(str).flatten())
        texts.extend(text_splitter.split_text(text))

    vectorstore = FAISS.from_texts(texts, embedding=embeddings)
    vectorstore.save_local(save_path)
    logger.info(f"Benefits database vectorstore created and saved to {save_path}")

def get_or_create_vectorstore(file_path, save_path):
    if os.path.exists(save_path):
        logger.info(f"Loading existing benefits database from {save_path}")
        return FAISS.load_local(save_path, OpenAIEmbeddings(openai_api_key=api_key), allow_dangerous_deserialization=True)
    else:
        logger.info("Creating new benefits database")
        create_and_save_vectorstore(file_path, save_path)
        return FAISS.load_local(save_path, OpenAIEmbeddings(openai_api_key=api_key), allow_dangerous_deserialization=True)

def update_vectorstore():
    global VectorStore
    logger.info("Updating benefits database")
    VectorStore = get_or_create_vectorstore("government_benefits.csv", "benefits_vectorstore.faiss")

# Initialize VectorStore
VectorStore = get_or_create_vectorstore("government_benefits.csv", "benefits_vectorstore.faiss")

# Schedule weekly updates to catch new programs
scheduler = BackgroundScheduler()
scheduler.add_job(func=update_vectorstore, trigger="interval", days=7)
scheduler.start()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/find_investors', methods=['POST'])  # keeping route name for compatibility
def find_investors():
    global VectorStore

    # Get user details from form
    age = request.form['age']
    income = request.form['income']
    household_size = request.form['householdSize']
    location = request.form['location']
    employment = request.form['employment']

    # Calculate some helpful derived values
    monthly_income = float(income) / 12
    federal_poverty_level = 13590 + (4720 * (int(household_size) - 1))  # 2023 FPL
    income_percentage_fpl = (float(income) / federal_poverty_level) * 100

    user_profile = f"""
    Age: {age}
    Annual Income: ${income}
    Monthly Income: ${monthly_income:.2f}
    Household Size: {household_size}
    Location: {location}
    Employment Status: {employment}
    Percentage of Federal Poverty Level: {income_percentage_fpl:.1f}%
    """

    prompt = f"""
    You are MyGovConnect, an expert AI assistant specializing in government benefits and assistance programs. Your mission is to help users find all government programs, grants, and services they may qualify for based on their profile:

    {user_profile}

    Analyze the following categories of assistance:
    1. Healthcare (e.g., Medicaid, CHIP, Medicare, ACA Subsidies)
    2. Food & Nutrition (e.g., SNAP, WIC, School Meals)
    3. Income Support (e.g., TANF, SSI, EITC)
    4. Housing (e.g., Section 8, Public Housing, LIHEAP)
    5. Education (e.g., FAFSA, Pell Grants, Head Start)
    6. Employment (e.g., Unemployment, Job Training, Workforce Programs)
    7. Senior & Disability Services (e.g., Social Security, Medicare, Paratransit)
    8. Child Care (e.g., CCDF, Head Start, After School Programs)

    For each matching program, provide:
    1. Program name and brief description
    2. Specific eligibility criteria met by the user
    3. Estimated benefit amount (if calculable)
    4. Application process and required documents
    5. Local office or website to apply
    6. Match confidence score (60-100%)

    Format each program as a JSON object with these fields:
    - "Investor Name": Program name
    - "Fund Focus Areas": Main category (Healthcare, Food, etc.)
    - "Location": Where program is available (State/County/City)
    - "Contact Details": Application info and URLs
    - "Likelihood to Invest": Match confidence score
    - "Match Reason": Detailed eligibility explanation

    Consider these factors:
    - Age-specific programs (child, adult, senior)
    - Income limits relative to Federal Poverty Level
    - Household size impacts on eligibility
    - State and local program variations
    - Employment status requirements
    - Emergency or temporary assistance options
    - Program combinations and interactions

    Prioritize:
    1. Programs with highest benefit value
    2. Emergency assistance if needed
    3. Programs with simplified application process
    4. Local programs in user's area
    5. Programs with highest match confidence

    Return only the JSON array without additional text. Include at least 10 highly relevant programs.
    """

    try:
        docs = VectorStore.similarity_search(query=prompt, k=7)  # Increased k for better coverage
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, request_timeout=300)  # Lower temperature for more consistent results
        chain = load_qa_chain(llm=llm, chain_type="stuff")

        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=prompt)

        logger.info(f"Successfully processed benefits search. Tokens used: {cb.total_tokens}")
        return jsonify({"investors": response})  # keeping response key for compatibility
    except Exception as e:
        logger.error(f"Error finding benefits: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred while searching for benefits"}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"An unhandled exception occurred: {str(e)}", exc_info=True)
    return jsonify({"error": "An internal server error occurred"}), 500

"""
if __name__ == '__main__':
    app.run(debug=True)
"""