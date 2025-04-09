# main.py
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever  # Now using the correct import

# Initialize LLM
model = Ollama(model="llama3.2", temperature=0.7)

# Prompt template
template = """You are an expert pizza restaurant assistant. 
Use these customer reviews to answer questions:
{reviews}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Interaction loop
print("\n🍕 Pizza Review Assistant (type 'q' to quit)")
print("----------------------------------------")

while True:
    question = input("\nYour question about our pizzas: ").strip()
    if question.lower() == 'q':
        break
    
    # Get relevant reviews
    reviews = retriever.invoke(question)
    print("\nFound relevant reviews:", reviews)
    
    # Generate answer
    response = chain.invoke({
        "reviews": "\n- " + "\n- ".join(reviews),
        "question": question
    })
    
    print("\nAssistant:")
    print("----------")
    print(response)