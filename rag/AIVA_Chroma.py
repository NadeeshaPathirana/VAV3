import time

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.llms import ChatMessage
from bs4 import BeautifulSoup
import requests
from llama_index.core import Document
from typing import List
import os
from llama_index.core.chat_engine import CondenseQuestionChatEngine
import json

from sentence_transformers import SentenceTransformer
from llama_index.core.embeddings import BaseEmbedding

class LocalHFEmbedding(BaseEmbedding):
    model: SentenceTransformer  # declare as field

    # Use a classmethod constructor instead of __init__ for offline model
    @classmethod
    def from_local_path(cls, model_path: str):
        model = SentenceTransformer(model_path)
        return cls(model=model)

    # Required sync embedding methods
    def _get_text_embedding(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

    def _get_query_embedding(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    # async version
    async def _aget_query_embedding(self, texts: List[str]) -> List[List[float]]:
        return self._get_query_embedding(texts)


# 3rd version of AIVA. Use Chroma DB instead of Qdrant

class AIVA_Chroma:
    def __init__(self, model="mistral"):
        self._chroma_client = chromadb.PersistentClient(path="./chroma_db")
        if self._chroma_client is None:
            raise RuntimeError("Failed to initialize ChromaDB PersistentClient.")
        self._llm = Ollama(model="mistral", request_timeout=240.0, temperature=0.3,)  # 240 seconds - increased to solve the timeout issue; Lower temperature reduces rambling.

        embed_model = LocalHFEmbedding(
            model=SentenceTransformer(
                r"C:\Users\220425722\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
            )
        )  # path to downloaded model

        self._service_context = ServiceContext.from_defaults(
            llm=self._llm,  # your local LLM object
            embed_model=embed_model
        )  # a lightweight Sentence Transformer model
        self._index = None
        self.model = model
        # self._api_key = 'AIzaSyAB_yU07EvwEc2D0pK8hJhoxjQZPwFUHxc'
        self._api_key = 'AIzaSyCWBlzpNEEgOkb3GsYdB3SDIOvUmr_h1ig'
        self._cse_id = '5086429ea12f641aa'
        self._big5_score = {"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5} # assuming 0.5 is the default value
        self._create_kb()
        self._create_chat_engine()

    def _create_chat_engine(self):
        self._memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        self._chat_engine = self._index.as_chat_engine(
            chat_mode="context",
            memory=self._memory,
            system_prompt=self._prompt,
            similarity_top_k=2
        )

    def _get_custom_prompt(self, emotion: str):
        custom_prompt = self._prompt
        if emotion == 'Happiness':
            emotion_user = 'happy'
            llm_response = 'happy tone'
        elif emotion == 'Anger':
            emotion_user = 'angry'
            llm_response = "calm tone"
        elif emotion == 'Sadness':
            emotion_user = 'sad'
            llm_response = 'sad tone and empathatically'
        else:
            emotion_user = 'neutral'
            llm_response = 'neutral tone'
        custom_prompt += "This person seems to be feeling " + emotion_user + " at the moment. Reply to the user in a " + llm_response + "Do not directly include the identified emotion in the conversation."
        return custom_prompt

    def load_profile_files(self, profile_dir: str):
        """
        Load and flatten all JSON and TXT files from a directory into LlamaIndex Documents.
        """
        documents = []

        for file_name in os.listdir(profile_dir):
            file_path = os.path.join(profile_dir, file_name)
            if not os.path.isfile(file_path):
                continue

            # --- JSON FILES ---
            if file_name.lower().endswith(".json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    def flatten_json(d, prefix=""):
                        items = []
                        for k, v in d.items():
                            if isinstance(v, dict):
                                items += flatten_json(v, f"{prefix}{k}.")
                            elif isinstance(v, list):
                                for i, val in enumerate(v):
                                    items.append(f"{prefix}{k}[{i}]: {val}")
                            else:
                                items.append(f"{prefix}{k}: {v}")
                        return items

                    flattened_texts = flatten_json(data)
                    for text in flattened_texts:
                        documents.append(Document(
                            text=text,
                            metadata={"source": file_name, "type": "profile_json"}
                        ))
                        if file_name == "user_profile.json":
                            self._create_personality_score_map(text)

                except Exception as e:
                    print(f"[WARN] Failed to load JSON {file_name}: {e}")

            else:
                print("Profile is not a json file")

        print(f"[INFO] Loaded {len(documents)} profile documents from {profile_dir}")
        return documents

    def _create_kb(self):
        try:
            profile_dir = r"C:\Users\220425722\Desktop\Python\VAV1\rag\profile"

            # Load and prepare documents
            documents = self.load_profile_files(profile_dir)

            # Ensure documents are not empty
            if not documents:
                raise ValueError("No documents found. Ensure the file exists and is not empty.")

            # Ensure ChromaDB client is initialized
            if self._chroma_client is None:
                raise RuntimeError("Failed to initialize ChromaDB PersistentClient.")

            # Ensure collection is created
            collection = self._chroma_client.get_or_create_collection("va_db")
            if collection is None:
                raise RuntimeError("Failed to create or retrieve ChromaDB collection.")

            vector_store = ChromaVectorStore(chroma_collection=collection, collection_name="va_db")

            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # Create the index
            self._index = VectorStoreIndex.from_documents(
                documents, service_context=self._service_context, storage_context=storage_context
            )

            # save index
            self._index.storage_context.persist(persist_dir="storage")  # TODO: write code to reuse the stored indices

            print("Knowledgebase created successfully using ChromaDB!")

        except Exception as e:
            print(f"Error while creating knowledgebase: {e}")
            self._index = None

    # def interact_with_llm(self, user_query, emotion=None, custom_prompt=None):
    #     start_time = time.time()
    #
    #     AgentChatResponse = self._chat_engine.chat(user_query)
    #     answer = AgentChatResponse.response
    #     end_time = time.time()  # End time measurement
    #     execution_time = end_time - start_time
    #     print(f"LLM Interaction Execution Time: {execution_time:.2f} seconds")  # Print the total execution time
    #     return answer

    def interact_with_llm(self, user_query, emotion=None, custom_prompt=None):
        start_time = time.time()

        try:

            memory_text = " ".join([m.content for m in self._memory.get_all()])  # get stored messages
            memory_token_count = len(memory_text.split())  # rough estimate (1 token ≈ 1 word)

            if memory_token_count > 1300:  # close to your 1500 limit
                print(f"[INFO] Memory near token limit ({memory_token_count} tokens). Summarizing...")
                summary_prompt = (
                    "Summarize the following conversation briefly in 3-4 sentences, keeping emotional tone:"
                )
                # Summarize memory using LLM (lightweight)
                summary_response = self._llm.complete(summary_prompt + "\n\n" + memory_text[:5000])
                summary = summary_response.text if hasattr(summary_response, "text") else str(summary_response) #todo: might need to save in the DB

                # Reset memory and insert summarized version
                self._memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
                self._memory.put(ChatMessage(role="system", content=f"Conversation summary: {summary}"))
                print("[INFO] Memory summarized successfully.")

        except Exception as e:
            print(f"[WARN] Memory summarization failed: {e}")

        try:
            # custom_prompt = self._get_custom_prompt(emotion)
            # custom_prompt = self._prompt
            #
            # dynamic_prompt = custom_prompt if custom_prompt else self._prompt
            #
            # self._chat_engine = self._index.as_chat_engine(
            #     chat_mode="context",
            #     memory=self._memory,
            #     system_prompt=dynamic_prompt,
            #     similarity_top_k=2,
            # )
            s_time = time.time()
            # agent_chat_response = self._chat_engine.chat("The user, Jane said: '" + user_query + "'. Reply to this appropriately") # todo: make this Jane said and pass user name in variable. If this is working, add the emotion here itself instead of the dynamic prompt
            agent_chat_response = self._chat_engine.chat(user_query)
            e_time = time.time()
            print(f"LLM Actual Interaction Time: {e_time - s_time:.2f} seconds")
            answer = agent_chat_response.response
        except Exception as e:
            print(f"[ERROR] Chat failed due to {e}. Resetting memory to recover.")
            self._memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
            answer = "I'm sorry, I had a little trouble remembering everything just now — but I’m back!"

        end_time = time.time()
        print(f"LLM Interaction Execution Time: {end_time - start_time:.2f} seconds")
        return answer

    def _get_personality_vs_com_style_query(self):
        output = "The following describes the older adult's personality and how to communicate effectively:\n\n. According to the available information, "
        if self._big5_score["openness"] >= 0.5:
            output += ("this person shows high openness. People who shows high openness, generally have the following "
                       "qualities: high questioningness, very creative, open to trying new things, focused on "
                       "tackling new challenges, happy to think about abstract concepts")
        elif self._big5_score["openness"] < 0.5:
            output = ("this person shows low openness. People who shows low openness, generally have the following "
                      "qualities: low questioningness,Dislikes change, Does not enjoy new things, Resists new ideas, "
                      "Not very imaginative, Dislikes abstract or  theoretical concepts")

        if self._big5_score["conscientiousness"] >= 0.5:
            output = ("this person shows high conscientiousness. People who shows high conscientiousness, generally "
                      "have the following qualities: high impression manipulativeness,spends time preparing, "
                      "finishes important tasks right away, pays attention to detail, enjoys having a set schedule")
        elif self._big5_score["conscientiousness"] < 0.5:
            output = ("this person shows low conscientiousness. People who shows low conscientiousness, generally "
                      "have the following qualities: low impression manipulativeness,Dislikes structure and  "
                      "schedules, Makes messes and doesn't  take care of things, Fails to return things or put  them "
                      "back where they  belong, Procrastinates important  tasks, Fails to complete necessary  or "
                      "assigned tasks")

        if self._big5_score["extraversion"] >= 0.5:
            output = ("this person shows high extraversion. People who shows high extraversion, generally have the "
                      "following qualities: more expressive in conversations,enjoys being the center of attention, "
                      "likes to start conversations, enjoys meeting new people, has a wide social circle of friends "
                      "and, finds it easy to make new friends, feels energized when around other people, say things "
                      "before thinking about them")
        elif self._big5_score["extraversion"] < 0.5:
            output = ("this person shows low extraversion. People who shows low extraversion, generally have the "
                      "following qualities: less expressive in conversations,Prefers solitude, Feels exhausted when  "
                      "having to socialize a lot, Finds it difficult to start  conversations, Dislikes making small "
                      "talk, Carefully thinks things  through before speaking, Dislikes being the center of  attention")

        if self._big5_score["agreeableness"] >= 0.5:
            output = ("this person shows high agreeableness. People who shows high agreeableness, generally have the "
                      "following qualities: has a great deal of interest in other people, cares about others, "
                      "feels empathy and concern for other people, enjoys helping and contributing to the happiness "
                      "of other people, assists others who are in need of help")
        elif self._big5_score["agreeableness"] < 0.5:
            output = ("this person shows low agreeableness. People who shows low agreeableness, generally have the "
                      "following qualities: Takes little interest in others, Doesn't care about how other people "
                      "feel, Has little interest in other people's problems, Insults and belittles others, "
                      "Manipulates others to get what they want")

        if self._big5_score["neuroticism"] >= 0.5:
            output = ("the person shows high neuroticism. People who shows high neuroticism, generally have the "
                      "following qualities: both emotionality and impression manipulativeness is high,Experiences a "
                      "lot of stress, Worries about many different things, Gets upset easily, Experiences dramatic "
                      "shifts in mood, Feels anxious, Struggles to bounce back after stressful events, "
                      "possible verbal aggressiveness")
        elif self._big5_score["neuroticism"] < 0.5:
            output = ("the person shows low neuroticism. People who shows low neuroticism, generally have the "
                      "following qualities: both emotionality and impression manipulativeness is low,Emotionally "
                      "stable, Deals well with stress, Rarely feels sad or depressed, Doesn't worry much, "
                      "Is very relaxed")

        output += ("When responding to the older adult, take into account both their personality traits described "
                   "above and their current emotional state. But, do not directly use these personality attributes to discribe the user in the conversation. \n\n")
        return output
    @property
    def _prompt(self):
        return ("""
         You are a warm, friendly, and attentive voice assistant designed to provide companionship and support to socially isolated older adults (a.k.a. user). Your name is cai. Your goal is to engage users in meaningful conversations. 
         
         <strict_rules>
            
            You may greet the user ONLY ONCE, at the beginning of the first message of the conversation, and only by using their name. Do NOT greet them again in later messages.
            
         </strict_rules>
         
         """
                # + self._get_personality_vs_com_style_query() + self._get_general_behaviour_query()
        )


    # def _prompt(self):
    #     return """
    #
    #         <priority_instructions>
    #             You must obey all rules in this prompt. Do not override or ignore them.
    #             Never generate long responses with more than 3 sentences.
    #             Do not greet the user in every utterance. Call the user by their name.
    #             Never skip or ignore a user question. If the user asks a question, you MUST provide a direct answer first, before anything else.
    #             Do not ask more than one question from the user at a time.
    #             Do not discuss multiple types of topics in a single utterance. ex: do not ask about hobbies while you are talking about family.
    #             Do not reveal the personality information in this prompt to the user.
    #             Do not override user information in the user profile.
    #             If you do not have information regarding a user's question, tell them you do not know that information.
    #             If the user introduces a topic, follow their lead and continue in that direction—unless it is a must, do not change the direction of the conversation on your own.
    #         </priority_instructions>
    #
    #         You are a warm, friendly, and attentive voice assistant named Cai designed to provide companionship and support to socially isolated older adults (a.k.a. user). Your goal is to engage them in meaningful conversations.
    #
    #         Keep in mind that women often communicate with greater agreeableness and expressiveness, while men tend to favor precision and conciseness. Adjust your tone and interaction style accordingly \n\n.
    #         """ + self._get_personality_vs_com_style_query() + self._get_general_behaviour_query()

    def _create_personality_score_map(self, text):
        if text.split(":")[0].strip() == 'personality_traits.openness_score':
            self._big5_score["openness"] = float(text.split(":")[1].strip())
        elif text.split(":")[0].strip() == 'personality_traits.conscientiousness_score':
            self._big5_score["conscientiousness"] = float(text.split(":")[1].strip())
        elif text.split(":")[0].strip() == 'personality_traits.extraversion_score':
            self._big5_score["extraversion"] = float(text.split(":")[1].strip())
        elif text.split(":")[0].strip() == 'personality_traits.agreeableness_score':
            self._big5_score["agreeableness"] = float(text.split(":")[1].strip())
        elif text.split(":")[0].strip() == 'personality_traits.neuroticism_score':
            self._big5_score["neuroticism"] = float(text.split(":")[1].strip())

    def _get_general_behaviour_query(self):
        output = """ Following topics will be appealing for the user:
        Childhood, Family and Life Events, Their Values and Identity, The Present,  Life Lessons, and Legacy, Past Experiences, Pets and Animals. Suggest one topic at a time. Avoid distressing topics. Ex: Disability, Death, Diseases, Losses, Fears. \n\n
        """
        return output
