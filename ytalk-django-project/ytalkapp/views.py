from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import secrets

# def hello(request):
#     return HttpResponse("Hello, World!")

from pytube import YouTube
import whisper
import os
from typing import Iterator, TextIO
from datetime import datetime
import pandas as pd


import re
import openai
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.chains import RetrievalQAWithSourcesChain

import faiss

if not os.environ.get('OPENAI_API_KEY'):
    os.environ['OPENAI_API_KEY'] = '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'
 


llm = OpenAI(temperature=0)

def home(request):
    return render(request, 'ytalkapp/index.html')

def get_video(_link, _path):
  try:
      print("called")
      yt = YouTube(_link)
  except:
      print("Connection Error")

  yt.streams.filter(file_extension='mp4')
  stream = yt.streams.get_by_itag(139)
  stream.download('',_path)
  title = yt.title
  print("video downloaded")
#   return True
  return title


def transcribe_me(_video, _translate=True):

  # setting options to define if translating or transcribing
  
  if _translate:
    _options = dict(task="translate", beam_size=5, best_of=5)
  else:
    _options = dict(task="transcribe", beam_size=5, best_of=5)
  model = whisper.load_model("base.en")  # for english the .en tend to perform better otherwise just use "base"
  result = model.transcribe(_video, **_options)

  return result, result["text"], result["segments"]


def format_timestamp(seconds: float):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    return (f"{hours}:" if hours > 0 else "") + f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def write_vtt(transcript: Iterator[dict], file: TextIO):
    print("WEBVTT\n", file=file)
    for segment in transcript:
        print(
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            f"{segment['text'].replace('-->', '->')}\n",
            file=file,
            flush=True,
        )

def slugify(title):
    return "".join(c if c.isalnum() else "_" for c in title).rstrip("_")


def gp3_summarize_new(_text, _chunk_size):
    openai.api_key = os.environ['OPENAI_API_KEY']
    #text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=_chunk_size, chunk_overlap=0)

    #text_splitter = SpacyTextSplitter(chunk_size=_chunk_size)
    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = _chunk_size,
    chunk_overlap  = 20,
    length_function = len,)

    texts = text_splitter.split_text(_text)
    docs = [Document(page_content=t) for t in texts[:15]]
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    out = chain.run(docs)
    return out.strip()

# extract the questions
def gp3_extract_questions(_text_chunks):
    openai.api_key = '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' # os.environ['OPENAI_API_KEY']

    texts = _text_chunks
    docs = [Document(page_content=t) for t in texts[:3]]

    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)
    query = "Extract the questions from this document"

    try:
      docs = docsearch.similarity_search(query)
    except Exception as e:
      print(e)

    chain_refine = load_qa_chain(OpenAI(temperature=0), chain_type="refine")
    out = chain_refine({"input_documents": docs, "question": query}, return_only_outputs=True)

    questions = out["output_text"].splitlines()
    questions = [q for q in questions if re.match(r"^\d+\.", q)]

    return questions

# split the transcription into segments and return text and timecodes
def store_segments(segments):
  texts = []
  start_times = []

  for segment in segments:
    text = segment['text']
    start = segment['start']

    # Convert the starting time to a datetime object
    start_datetime = datetime.fromtimestamp(start)

    # Format the starting time as a string in the format "00:00:00"
    formatted_start_time = start_datetime.strftime('%H:%M:%S')

    texts.append("".join(text))
    start_times.append(formatted_start_time)

  return texts, start_times

# find the segment that answer best the question
def answer_question(_texts, _start_times, _question):

  text_splitter = CharacterTextSplitter(chunk_size=1500)
  docs = []
  metadatas = []
  for i, d in enumerate(_texts):

    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": _start_times[i]}] * len(splits))
    embeddings = OpenAIEmbeddings()

  store = FAISS.from_texts(docs, embeddings, metadatas=metadatas)
  faiss.write_index(store.index, "docs.index")

  chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=store)
  result = chain({"question": _question})

  answer = result['answer']
  sources = result['sources']

  return answer, sources

# create a link that directs the user to the provided timecode
def add_timecode_to_url(url, timecode):
    # convert timecode to seconds
    time_parts = timecode.split(':')
    seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
    # append the timecode to the URL
    return url + "&t=" + str(seconds) + "s"

def get_output_df(questions, texts, start_times):
    data = []
    for q in questions:
        try:
            r = answer_question(texts, start_times, q)
            timecodes = r[1].split(",")
            urls = []
            for t in timecodes:
                urls.append(add_timecode_to_url(link, t))
            data.append([q, r[0], r[1], urls])
        except Exception as e:
            pass
    df = pd.DataFrame(data, columns=['Question', 'Answer', 'Timecodes', 'URLs'])
    return df


def get_video(_link, _path):
  try:
      print("called")
      yt = YouTube(_link)
  except:
      print("Connection Error")

  yt.streams.filter(file_extension='mp4')
  stream = yt.streams.get_by_itag(139)
  stream.download('',_path)
  title = yt.title
  print("video downloaded")
#   return True
  return title


def transcribe_me(_video, _translate=True):
  # setting options to define if translating or transcribing
  if _translate:
    _options = dict(task="translate", beam_size=5, best_of=5)
  else:
    _options = dict(task="transcribe", beam_size=5, best_of=5)
  model = whisper.load_model("base.en")  # for english the .en tend to perform better otherwise just use "base"
  result = model.transcribe(_video, **_options)

  return result, result["text"], result["segments"]

def generate_random_str(length):
  return ''.join(secrets.choice('abcdefghijklmnopqrstuvwxyz0123456789') for i in range(length))

@csrf_exempt
def post_link(request):
    print("I am here ")
    if request.method == 'POST':
        link = request.POST.get('link', None)
        question = request.POST.get('question', None)# get the link parameter from POST data
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_path = f"{dir_path}/mp4files/video-{generate_random_str(3)}.mp4"
        if link and question:
          title = get_video(link, file_path)
          # if not os.path.isfile(vtt_path):
          #   output, extracted_text, res  = transcribe_me(file_path)
          # title = "Miyamoto Musashi - How to Build Self-Discipline"
          output, extracted_text, res  = transcribe_me(file_path)
          vtt_path = f"{dir_path}/vttfiles/{slugify(title)}.vtt"
          # if not os.path.isfile(vtt_path):
          with open(vtt_path, 'w', encoding="utf-8") as vtt:
            write_vtt(output["segments"], file=vtt)
            #os.path.join(".", f"{slugify(title)}.vtt")
          
            # print saved message with absolute path
          print("Saved VTT to", os.path.abspath(vtt_path))
          summary = gp3_summarize_new(extracted_text, 1600)
          print(f"summary: {summary}")
          # we store text and timecodes
          texts, start_times = store_segments(res)
          # here is the list of questions
          questions = gp3_extract_questions(texts)
          print(f"questions: {questions} ")
          r = answer_question(texts, start_times, question)
          print(question)
          print(r[0])
          timecodes = r[1].split(",")
          print(r[1])
          all_links=list()
          
          for t in timecodes:
            all_links.append(add_timecode_to_url(link, t))
            # print(add_timecode_to_url(link, t))
 
          context = {'summary': summary, 'links' : all_links}
          return render(request, 'ytalkapp/response.html', context) 
    
        else:
            return JsonResponse({'error': 'No link provided.'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request.'}, status=405)



