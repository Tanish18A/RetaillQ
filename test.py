from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import dotenv

dotenv.load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash"
)

parser = StrOutputParser()

chain = llm | parser

response = chain.invoke("What is AI?")

print(response.content)