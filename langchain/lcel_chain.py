# 1. Import LCEL building blocks
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda
)

# 2. Prompt (Runnable)
prompt = ChatPromptTemplate.from_template("""
You are a senior AI engineer.

Explain the following concept clearly with an example:

{question}
""")

# 3. Ollama LLM(Streaming + Async)
llm  = ChatOllama(
    model= "llama3.2:3b",
    temperature=0.2,
    streaming=True
)

# 4. Output Parser
parser = StrOutputParser()

# 5. Compose Using LCEL Pipe
chain = (
    {"question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

# 6. STREAMING EXAMPLE (Chat UI / CLI)
print("Streaming output:\n")
for token in chain.stream("What is LCEL in langchain?"):
    print (token, end="", flush=True)

# 7. ASYNC EXAMPLE
# import asyncio

# async def run_async():
#     result = await chain.ainvoke(
#         "Why is async important in LLM systems?"
#     )
#     print("\n\nAsync result:\n", result)

# asyncio.run(run_async())

# 8. TRACING (YES, EVEN WITH OLLAMA)
# export LANGCHAIN_TRACING_V2=false
# export LANGCHAIN_PROJECT="ollama-lcel-demo"

# 9. COMPOSABILITY (Add Logic Without Rewrite)
def safety_filter(text: str) -> str:
    if "violence" in text.lower():
        return "⚠️ Content blocked by safety policy"
    return text

safety_runnable = RunnableLambda(safety_filter)
safe_chain = chain | safety_runnable

# 10. FINAL PRODUCTION LCEL PIPELINE
final_chain = (
    {"question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
    | safety_runnable
)