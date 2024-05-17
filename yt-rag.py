from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain.prompts import MessagesPlaceholder ,ChatPromptTemplate
from langchain.chains import create_retrieval_chain,create_history_aware_retriever
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os,re 
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()


# extract the youtube id 
def extract_video_id(url):
    pattern = r'(?<=v=)[a-zA-Z0-9_-]+'
    match = re.search(pattern, url)
    if match:
        return match.group(0)
    else:
        return None
    
# use the id to get the transcript with the timestamps
def get_transcript(id):
    transcript = YouTubeTranscriptApi.get_transcript(id)
    transcript_text = ''
    for segment in transcript:
        start = segment['start']
        duration = segment['start']+segment['duration']
        text = segment['text']
        start_hour, start_remainder = divmod(start, 3600)
        start_min, start_sec = divmod(start_remainder, 60)
        
        end_hour, end_remainder = divmod(duration, 3600)
        end_min, end_sec = divmod(end_remainder, 60)
        
        transcript_text += f"{int(start_hour):02d}:{int(start_min):02d}:{int(start_sec):02d} --> {int(end_hour):02d}:{int(end_min):02d}:{int(end_sec):02d}: {text}\n"
    return transcript_text




# create rag chain that answer our queries 
def create_chain(transcript_text):


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100,add_start_index=True)
    splits = text_splitter.split_text(transcript_text)
    db =Chroma.from_texts(embedding=OpenAIEmbeddings(),texts=splits)

    retriever = db.as_retriever()

    conceptalize_prompt = ChatPromptTemplate.from_messages([
        ('system',"Given a chat history and the latest user question, create a standalone question that doesn't require knowledge of the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    
    ])
    chat = ChatOpenAI(model='gpt-3.5-turbo-0125', temperature=0)
    history_aware_retriever = create_history_aware_retriever(llm=chat,retriever=retriever,prompt=conceptalize_prompt)
    qa_system_prompt = '''
    You're a question-answering assistant. Use the provided context to answer questions.Provide as much detail in your answers.If you're unsure, say so.
    {context}
    '''
    prompt = ChatPromptTemplate.from_messages([
        ('system',qa_system_prompt),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human','{input}')
    ])
    chain = create_stuff_documents_chain(llm=chat,prompt=prompt)
    rag_chain = create_retrieval_chain(combine_docs_chain=chain,retriever=history_aware_retriever)
    return rag_chain

if __name__=="__main__" :
    chat_history = []
    question = input('what is your question? ')
    id = extract_video_id('https://www.youtube.com/watch?v=bzbsJGMVHxQ')
    transcript = get_transcript(id)
    chain = create_chain(transcript)

    while question != 'q':
        result = chain.invoke({'input':question,'chat_history':chat_history})
        chat_history.extend([HumanMessage(content=question),result['answer']]) 
        print(result['answer'])

        question = input('what is your question? ')


