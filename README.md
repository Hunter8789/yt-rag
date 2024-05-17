# yt-rag
This is package is for question and answering from Youtube videos. It strives to give very good answers, with no hallucinations. 
By default, it uses OpenAI and Chroma, but Claude and Faiss would also work well. The package also utlizes the langchain framework to develops its chains.

OBJECTIVE:
The goal for this package is to find the desired content and its timestamp in the video since some videos do not provide chapters. 

Example:
link : https://www.youtube.com/watch?v=yfHHvmaMkcA

Question: what time does she start talking about deploying vector databases

Answer: Chatbot: She starts talking about deploying vector databases at 00:15:58.

![Screenshot 2024-05-17 at 3 13 48 PM](https://github.com/Hunter8789/yt-rag/assets/106272424/0ef4e2e5-ec92-4f90-97e7-5df8d426f7dc)


Notes:

The video url must have the video id like this  v=yfHHvmaMkcA. 
The original goal is to find content and its timestamps, but I realized this can be great tool for getting video summaries. 
The code stores the vectors of the transcript on the local disk using Chromadb. 
