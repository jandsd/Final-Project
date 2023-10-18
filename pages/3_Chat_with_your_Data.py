# import os
# import utils
# import streamlit as st
# from streaming import StreamHandler

# from langchain.chat_models import ChatOpenAI
# from langchain.document_loaders import PyPDFLoader
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# st.set_page_config(page_title="ChatPDF", page_icon="📄")
# st.header('Chat with your Documents')
# st.write('Has access to custom documents and can respond to user queries by referring to the content within those documents')

# class CustomDataChatbot:

#     def __init__(self):
#         utils.configure_openai_api_key()
#         self.openai_model = "gpt-3.5-turbo"
#         self.qa_chain = None  # Initialize the QA chain

#     def save_file(self, file):
#         folder = 'tmp'
#         if not os.path.exists(folder):
#             os.makedirs(folder)

#         file_path = f'./{folder}/{file.name}'
#         with open(file_path, 'wb') as f:
#             f.write(file.getvalue())
#         return file_path

#     def process_documents(self, uploaded_files):
#         # Load documents
#         docs = []

#         for file in uploaded_files:
#             file_path = self.save_file(file)
#             loader = PyPDFLoader(file_path)
#             docs.extend(loader.load())

#         # Split documents
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1500,
#             chunk_overlap=200
#         )
#         splits = text_splitter.split_documents(docs)

#         # Create embeddings and store in vectordb
#         embeddings = OpenAIEmbeddings()
#         vectordb = FAISS.from_documents(splits, embeddings)

#         # Define retriever
#         retriever = vectordb.as_retriever()

#         return retriever

#     @st.spinner('Analyzing documents..')
#     def setup_qa_chain(self, uploaded_files):
#         # if self.qa_chain is None:
#         retriever = self.process_documents(uploaded_files)
        
#         # Setup memory for contextual conversation
#         memory = ConversationBufferMemory(
#             memory_key='chat_history',
#             return_messages=True
#         )

#         # Setup LLM and QA chain
#         llm = ChatOpenAI(model_name=self.openai_model,
#                             temperature=0, streaming=True)
#         self.qa_chain = ConversationalRetrievalChain.from_llm(
#             llm, retriever=retriever, memory=memory, verbose=True)
#         # self.qa_chain = qa_chain
#         # return self.qa_chain

#     @utils.enable_chat_history
#     def main(self):
#         # User Inputs
#         uploaded_files = st.sidebar.file_uploader(label='Upload PDF files', type=[
#                                                   'pdf'], accept_multiple_files=True)
#         process_button = st.sidebar.button("Process Documents")
        
#         if process_button:
#             if uploaded_files:
#                 self.setup_qa_chain(uploaded_files)
#                 st.success("Documents processed and retriever set up!")

#             if not uploaded_files:
#                 st.error("Please upload PDF documents to continue!")
#                 st.stop()

#         user_query = st.chat_input(placeholder="Ask me anything!")
#         # print(self.qa_chain)

#         if user_query:
#             if self.qa_chain is None:
#                 st.error("Please process the documents before making queries!")
#             else:
#                 utils.display_msg(user_query, 'user')

#                 with st.chat_message("assistant"):
#                     st_cb = StreamHandler(st.empty())
#                     response = self.qa_chain.run(user_query, callbacks=[st_cb])
#                     st.session_state.messages.append(
#                         {"role": "assistant", "content": response})

# if __name__ == "__main__":
#     obj = CustomDataChatbot()
#     obj.main()



import os
import utils
import streamlit as st
from streaming import StreamHandler

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title="ChatPDF", page_icon="📄")
st.header('Chat with your Documents')
st.write('Has access to custom documents and can respond to user queries by referring to the content within those documents')

class CustomDataChatbot:

    def __init__(self):
        utils.configure_openai_api_key()
        self.openai_model = "gpt-3.5-turbo"
        self.qa_chain = None  # Initialize the QA chain
        self.vectordb = None

    def save_file(self, file):
        folder = 'tmp'
        if not os.path.exists(folder):
            os.makedirs(folder)

        file_path = f'./{folder}/{file.name}'
        with open(file_path, 'wb') as f:
            f.write(file.getvalue())
        return file_path

    def process_documents(self, uploaded_files):
        # Load documents
        docs = []

        for file in uploaded_files:
            file_path = self.save_file(file)
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        # Create embeddings and store in vectordb
        embeddings = OpenAIEmbeddings()
        self.vectordb = FAISS.from_documents(splits, embeddings)

        # Define retriever
        # retriever = vectordb.as_retriever()

        # return vectordb

    @st.spinner('Analyzing documents..')
    def setup_qa_chain(self, uploaded_files):
        self.process_documents(uploaded_files)
        
        # Setup memory for contextual conversation
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )

        # Setup LLM and QA chain
        llm = ChatOpenAI(model_name=self.openai_model,
                            temperature=0, streaming=True)
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm, retriever=self.vectordb.as_retriever(), memory=memory, verbose=True)

    @utils.enable_chat_history
    def main(self):
        # Use st.session_state to retain self.qa_chain
        if 'qa_chain' not in st.session_state:
            st.session_state.qa_chain = None

        # User Inputs
        uploaded_files = st.sidebar.file_uploader(label='Upload PDF files', type=[
                                                  'pdf'], accept_multiple_files=True)
        process_button = st.sidebar.button("Process Documents")
        
        if process_button:
            if uploaded_files:
                self.setup_qa_chain(uploaded_files)
                st.session_state.qa_chain = self.qa_chain  # Store the QA chain in session_state
                st.success("Documents processed and retriever set up!")

            if not uploaded_files:
                st.error("Please upload PDF documents to continue!")
                st.stop()

        user_query = st.chat_input(placeholder="Ask me anything!")

        if user_query:
            if st.session_state.qa_chain is None:
                st.error("Please process the documents before making queries!")
            else:
                utils.display_msg(user_query, 'user')

                with st.chat_message("assistant"):
                    st_cb = StreamHandler(st.empty())
                    response = st.session_state.qa_chain.run(user_query, callbacks=[st_cb])
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response})

if __name__ == "__main__":
    obj = CustomDataChatbot()
    obj.main()
