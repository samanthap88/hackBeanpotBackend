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
    logger.info(f"VectorStore created and saved to {save_path}")


def get_or_create_vectorstore(file_path, save_path):
    if os.path.exists(save_path):
        logger.info(f"Loading existing VectorStore from {save_path}")
        return FAISS.load_local(save_path, OpenAIEmbeddings(openai_api_key=api_key), allow_dangerous_deserialization=True)
    else:
        logger.info("Creating new VectorStore")
        create_and_save_vectorstore(file_path, save_path)
        return FAISS.load_local(save_path, OpenAIEmbeddings(openai_api_key=api_key), allow_dangerous_deserialization=True)


def update_vectorstore():
    global VectorStore
    logger.info("Updating VectorStore")
    VectorStore = get_or_create_vectorstore("maininvestorbase.csv", "investor_vectorstore.faiss")


# Initialize VectorStore
VectorStore = get_or_create_vectorstore("maininvestorbase.csv", "investor_vectorstore.faiss")

# Schedule periodic updates
scheduler = BackgroundScheduler()
scheduler.add_job(func=update_vectorstore, trigger="interval", days=7)
scheduler.start()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/find_investors', methods=['POST'])
def find_investors():
    global VectorStore

    industry = request.form['industry']
    stage = request.form['stage']
    description = request.form['description']
    location = request.form['location']

    startup_details = f"""
    Industry: {industry}
    Funding Stage: {stage}
    Description: {description}
    Location: {location}
    """

    json_structure = """
    [
    {
      "Investor Name": "string",
      "Fund Focus Areas": "string",
      "Location": "string",
      "Contact Details": {
        "Partner Name": "string",
        "Email": "string",
        "Website": "string",
        "Social Links": {
          "Twitter": "string",
          "LinkedIn": "string",
          "Facebook": "string"
        }
      },
      "Likelihood to Invest": "number %",
      "Match Reason": "string"
    }
    ]
    """

    prompt = f"""
    As an AI-powered investor matching system, your task is to analyze the given startup information and provide a list of potential investors from the database who are most likely to invest in this startup. The startup details are as follows:

    {startup_details}

    Using the following investor data fields to evaluate matches:
    - Investor Name
    - Fund Type
    - Website (if available)
    - Fund Focus (Sectors)
    - Partner Name
    - Partner Email
    - Location
    - Twitter Link
    - LinkedIn Link
    - Facebook Link

    You will use the following internal fields for context but do not display them to the user:
    - Fund Description
    - Portfolio Companies
    - Number of Investments
    - Number of Exits
    - Preferred Investment Stages

    Evaluate the following key criteria when selecting potential investors:
    1. **Industry alignment**: Ensure that the investor's fund focus (sectors) aligns with the startup's industry and market niche.
    2. **Investment stage**: Match the startup's funding stage with the investor's preferred fund stage.
    3. **Geographic proximity**: Consider location relevance, favoring investors who are in or focus on regions near the startup's location, but do not exclude global opportunities where geography is not a limiting factor.
    4. **Portfolio companies fit**: If available, assess whether the startup aligns with the types of companies already in the investor's portfolio (similar markets, technologies, or sectors).
    5. **Investment thesis alignment**: Look at the fund description to ensure that the investor's philosophy or thesis aligns with the startup's vision or mission, and explain why this investor would be a strategic match.

    Based on this data, provide a list of a minimum of **10-15 investors** who would be the best match for this startup. Return the result in the following JSON format for each investor:

    {json_structure}

    The match reason should prioritize the startup description when matching with an investor.

    Please provide the results directly in JSON format without any additional explanations and make sure to start and end with square brackets.
    """

    try:
        docs = VectorStore.similarity_search(query=prompt, k=5)
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, request_timeout=300)
        chain = load_qa_chain(llm=llm, chain_type="stuff")

        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=prompt)

        logger.info(f"Successfully processed request. Tokens used: {cb.total_tokens}")
        return jsonify({"investors": response})
    except Exception as e:
        logger.error(f"An error occurred while processing the request: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred while processing your request"}), 500


@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"An unhandled exception occurred: {str(e)}", exc_info=True)
    return jsonify({"error": "An internal server error occurred"}), 500

"""
if __name__ == '__main__':
    app.run(debug=True)
"""