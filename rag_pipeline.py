from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder
import streamlit as st

from config import (GROQ_MODEL,TOP_K_RESULTS)
from config import (GROQ_MODEL,INITIAL_RETRIEVAL_K,FINAL_TOP_K,RERANKER_MODEL)

llm_model = ChatGroq(model=GROQ_MODEL,temperature=0)

@st.cache_resource
def load_reranker():
    return CrossEncoder(RERANKER_MODEL)

reranker = load_reranker()

RAG_PROMPT = """
You are an AI Legal Assistant specialized in answering questions ONLY from uploaded legal documents.

STRICT INSTRUCTIONS:
1. Use ONLY the provided context to answer.
2. Do NOT use external knowledge, assumptions, or prior training.
3. If the answer is not explicitly present in the context, respond EXACTLY:
   "I don't know based on the uploaded documents."
4. Never fabricate legal interpretations or laws.
5. If applicable, mention the relevant Article, Section, Clause, or legal reference found in the context.
6. For rights-related questions:
   - Identify the relevant right/article.
   - Briefly explain why it applies using the provided context.
7. If the question is ambiguous, answer using only the closest relevant context.
8. Keep responses concise, factual, and legally grounded.

RESPONSE FORMAT:
- Relevant Article/Section: <if available>
- Answer: <concise explanation based on context>

Question:
{question}

Retrieved Context:
{context}

Answer:
"""

# RRF done after retrieval from both db
def reciprocal_rank_fusion(results,k=8):
    fused_scores={}
    unique_docs={}

    for docs in results:
        for rank, doc in enumerate(docs):

            doc_id=doc.page_content
            unique_docs[doc_id]=doc

            if doc_id not in fused_scores:
                fused_scores[doc_id]=0
            
            fused_scores[doc_id]+=1/(k+rank+1)
    
    reranked_results=sorted(
        fused_scores.items(),
        key=lambda x:x[1],
        reverse=True
    )
    final_docs= [
        unique_docs[doc_id]
        for doc_id, _ in reranked_results
    ]

    return final_docs


# AFTER RRF DO RERANK 
def rerank_documents(query, documents):

    pairs = [ [query, doc.page_content] for doc in documents]

    scores = reranker.predict(pairs)
    scored_docs = list(zip(documents, scores))

    reranked_docs = sorted(
        scored_docs,
        key=lambda x: x[1],
        reverse=True
    )

    final_docs = [doc for doc, score in reranked_docs ]

    return final_docs


# RETRIEVE RELEVENT DOCUMENTS
def retrieve_documents(query, retrievers):

    vector_db = retrievers["vector_db"]
    bm25_retriever = retrievers["bm25_retriever"]

    vector_retriever = vector_db.as_retriever(
        search_kwargs={"k": TOP_K_RESULTS}
    )

    vector_docs = vector_retriever.invoke(query)

    bm25_docs = bm25_retriever.invoke(query)

    fused_docs = reciprocal_rank_fusion(
        [vector_docs, bm25_docs]
    )
    
    # return fused_docs[:TOP_K_RESULTS]
    # print(f"before_rerank:",len(fused_docs[:TOP_K_RESULTS]))
    # for i, doc in enumerate(fused_docs):
    #     print(f"{i+1}. {doc.page_content[:200]}")
    # print("*"*50)



    reranked_docs = rerank_documents(
        query,
        fused_docs
    )

    # print("\nRETRIEVED DOCS:")
    # for i, doc in enumerate(reranked_docs):
    #     print(f"{i+1}. {doc.page_content[:200]}")


    return reranked_docs[:FINAL_TOP_K]



# GET CONTEXT
def get_context(documents):
    context = "\n\n".join(
        [doc.page_content for doc in documents]
    )

    return context


# ANSWER QUERY
def answer_query(query, retrievers):

    documents = retrieve_documents(query,retrievers)

    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

    chain = prompt | llm_model

    response = chain.invoke({
        "question": query,
        "context": context
    })

    return {
        "answer": response.content,
        "documents": documents
    }