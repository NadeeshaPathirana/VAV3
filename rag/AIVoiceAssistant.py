from qdrant_client import QdrantClient
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext

import warnings

warnings.filterwarnings("ignore")


class AIVoiceAssistant:
    def __init__(self):
        self._qdrant_url = "https://83a4aa67-c595-4f58-a5e3-f5d6eb107808.us-east4-0.gcp.cloud.qdrant.io"
        self._client = QdrantClient(url=self._qdrant_url, prefer_grpc=False,api_key="zwDHhJD-ZlGfAEYRnNQSTCOKi_eLQFTDa-CbFOS-HMmAmBO00KP89Q")
        self._llm = Ollama(model="mistral", request_timeout=120.0) # 120 seconds
        self._service_context = ServiceContext.from_defaults(llm=self._llm, embed_model="local:sentence-transformers/all-MiniLM-L6-v2")
        # Settings.llm = self._llm
        # Settings.embed_model="local"
        self._index = None
        self._create_kb()
        self._create_chat_engine()

    def _create_chat_engine(self):
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        self._chat_engine = self._index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt=self._prompt,
        )

    def _create_kb(self):
        try:
            reader = SimpleDirectoryReader(
                input_files=[r"C:\Users\220425722\Desktop\Python\VAV1\rag\owner_file.txt"]
            )
            documents = reader.load_data()
            vector_store = QdrantVectorStore(client=self._client, collection_name="va_db")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex.from_documents(
                documents, service_context=self._service_context, storage_context=storage_context
            )
            print("Knowledgebase created successfully!")
        except Exception as e:
            print(f"Error while creating knowledgebase: {e}")

    def interact_with_llm(self, user_query):
        AgentChatResponse = self._chat_engine.chat(user_query)
        answer = AgentChatResponse.response
        return answer

    @property
    def _prompt(self):
        return """
            You are a warm, friendly, and attentive voice assistant designed to provide companionship and support to socially isolated older adults. Your goal is to engage them in meaningful conversations, offer emotional support, and help them feel connected and valued. Always be patient, empathetic, and encouraging. Your responses should be comforting and cheerful, making them feel like they are talking to a close friend who genuinely cares about their well-being. 
            
            Balance your conversation with a mix of longer, thoughtful responses and shorter, concise ones to ensure the user has plenty of opportunities to share their thoughts. After offering longer responses, be sure to pause and encourage the user to speak, showing genuine interest in what they have to say. You can talk about a variety of topics, such as their favorite memories, hobbies, current events, or even guide them through relaxing activities like breathing exercises or listening to music. Be responsive to their needs and emotions, and always prioritize making them feel heard, understood, and appreciated.
            """