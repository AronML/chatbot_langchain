
import streamlit as st
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.llms import HuggingFaceHub
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chat_models import ChatLiteLLM
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template


def split_search_wiki(topic):
    docs = WikipediaLoader(query=topic, load_max_docs=10).load()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size = 100, chunk_overlap = 0
    )
    texts = text_splitter.split_text(docs[0].page_content)
    return texts

def vector_store(chunks):
    embeddings_model = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings_model)
    return vectorstore

def get_contextual_retriever(vectorstore, query):
    llm = HuggingFaceHub(repo_id = "google/flan-t5-xxl", model_kwargs = {"temperature": 0.5, "max_length": 512})
    retriever = vectorstore.as_retriever()
    compressor = LLMChainExtractor.from_llm(llm)
    compressor_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
    docs_comp = compressor_retriever.get_relevant_documents(query)
    return docs_comp[0].page_content

def conversation_history_search(contextual_retriever, topic, query):
    st.session_state.conversation.memory.save_context({"input":topic +"\n"+query}, {"output": contextual_retriever})
    
   
def handle_userinput():
    
    for i in range(len(st.session_state.conversation.memory.chat_memory.messages)):
        
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", st.session_state.conversation.memory.chat_memory.messages[i].content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", st.session_state.conversation.memory.chat_memory.messages[i].content), unsafe_allow_html=True)

def main():

    load_dotenv()
    chat_model = ChatLiteLLM(model="huggingface/codellama/CodeLlama-34b-Instruct-hf")  
    conversation = ConversationChain(llm=chat_model,memory = ConversationBufferMemory())


    st.set_page_config(page_title="ChatBot",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = conversation
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("ChatBot")
    user_question = st.text_input("Ask me anything:")
    
    if st.button("send"):
        st.session_state.conversation.predict(input=user_question)
        
        

    with st.sidebar:
        st.subheader("Research in Wikipedia")
        topic = st.text_input("topic")
        query = st.text_input("ask me something about the topic")
        if st.button("search"):
            with st.spinner("Processing"):
                # get pdf text
                sptd_search = split_search_wiki(topic)

                # get the text chunks
                vectorstore = vector_store(sptd_search)

                # create vector store
                contextual_retriever = get_contextual_retriever(vectorstore,query)

                # create conversation chain
                conversation_history_search(contextual_retriever, topic, query)
    if len(st.session_state.conversation.memory.chat_memory.messages) > 0:
        handle_userinput()
   
if __name__ == '__main__':
    main()