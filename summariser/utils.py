import requests
from decouple import config
import os
import wave
import time
import pickle
import warnings
import numpy as np
from sklearn import preprocessing
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn.mixture import GaussianMixture 
from nltk.tokenize import sent_tokenize
import glob
from django.conf import settings
import string

from .postag import get_postagged_transcript
from .para_generator import predict_dial_tags

errors = list()
def read_file(filename, chunk_size=5242880):
     with open(filename, 'rb') as _file:
         while True:
             data = _file.read(chunk_size)
             if not data:
                 break
             yield data

def assemblyai_transcript(filename):
  headers = {'authorization': "6997385c5df24e9294928acd98b8a0b9"}
  t_headers = {'authorization': "6997385c5df24e9294928acd98b8a0b9", 'content-type': "application/json"}
  
  response = requests.post('https://api.assemblyai.com/v2/upload', headers=headers, data=read_file(filename))
  
  upload_url = response.json()['upload_url']
  response = requests.post("https://api.assemblyai.com/v2/transcript", 
                           json={"audio_url": upload_url}, headers=t_headers)

  transcript_id = response.json()['id']

  return transcript_id

def get_aaitranscript(id):
  headers = {'authorization': "6997385c5df24e9294928acd98b8a0b9"}
  response = requests.get(f"https://api.assemblyai.com/v2/transcript/{id}", headers=headers)

  status = response.json()['status']
  while (status != 'completed'):
    if (status == 'error'):
      errors.append(response)
      print("Error Occurred !!")
      return "" 
    response = requests.get(f"https://api.assemblyai.com/v2/transcript/{id}", headers=headers)
    status = response.json()['status']

  return response.json()['text']

# speaker identification functions
def calculate_delta(array):
    rows,cols = array.shape
    #print(rows)
    #print(cols)
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
              first =0
            else:
              first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j 
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas


def extract_features(audio,rate):  
    mfcc_feature = mfcc.mfcc(audio,rate, 0.025, 0.01,20,nfft = 1200, appendEnergy = True)    
    mfcc_feature = preprocessing.scale(mfcc_feature)
    # print(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature,delta)) 
    return combined

def get_speakeridentity(filename):
  sourcepath = filename  
  modelpath = "simodels/"
    
  gmm_files = [os.path.join(modelpath, fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]
    
  #Load the Gaussian gender Models
  models = [pickle.load(open(fname,'rb')) for fname in gmm_files]
  speakers = [fname.split("/")[-1].split(".gmm")[0] for fname in gmm_files]

  # print(sourcepath)
  sr, audio = read(sourcepath)
  vector = extract_features(audio,sr)
    
  log_likelihood = np.zeros(len(models)) 

  for i in range(len(models)):
      gmm = models[i]  #checking with each model one by one
      scores = np.array(gmm.score(vector))
      log_likelihood[i] = scores.sum()
    
  # print(f"Confidence Scores: {log_likelihood}")
  winner = np.argmax(log_likelihood)
  # print(speakers)
  # print(f"Detected as - {speakers[winner]}")
  return speakers[winner]

def convert_sentences(text): 
  sentences =[]        
  sentences = sent_tokenize(text)    
  for sentence in sentences:        
    sentence.replace("[^a-zA-Z0-9]"," ")     
  return sentences

def preprocess_transcript(transcript):
  trans = transcript.split("\n\n")
  speakerwise_trans = dict()
  for t in trans:
    c = t.split(": ")
    if c[0] not in speakerwise_trans:
      speakerwise_trans[c[0]] = c[1]
    else:
      speakerwise_trans[c[0]] += c[1]

  for k in speakerwise_trans.keys():
    text = speakerwise_trans[k]
    text = text.replace('.', '. ').replace('?', '? ').replace('!', '! ')
    text = text.replace('  ', ' ') 
    speakerwise_trans[k] = text
  return speakerwise_trans

def get_esummary(speakerwise_trans):
  esumm = dict()
  for k in speakerwise_trans.keys():
    sentences = convert_sentences(speakerwise_trans[k])
    if len(sentences) < 4:
      result = settings.MODEL_SBERT(speakerwise_trans[k], num_sentences=len(sentences))
    else:
      result = settings.MODEL_SBERT(speakerwise_trans[k], num_sentences=4)
    esumm[k] = []
    for sen in convert_sentences(result):
      esumm[k].append(sen)
  return esumm

def esumm_to_str(esummary):
  esumm = list()
  for k in esummary.keys():
    es_sen = '\n'.join(esummary[k])
    es = f"{k}: \n\n{es_sen}"
    esumm.append(es)

  esumm_str = '\n\n'.join(esumm) 
  return esumm_str

##### ABSTRACTIVE FUNCTIONS

def split_conv_sentences(trans_list):
  newtrans_list = []
  for t in trans_list:
    info = t.split(": ")
    sentences = convert_sentences(info[1])
    for sen in sentences:
      newtrans_list.append(f"{info[0]}:{sen}")

  newtrans_list.insert(0, "speaker:pos")
  return "\n".join(newtrans_list)

def remove_unwanted_spaces(text):
    # Remove leading and trailing whitespaces
    text = text.strip()

    # Remove spaces before and after punctuations
    for punctuation in string.punctuation:
        text = text.replace(" " + punctuation, punctuation)
        text = text.replace(punctuation + " ", punctuation)

    return text

def get_asummary(transcript):
  # convert transcript to paragraph
  transcript = split_conv_sentences(transcript)
  postagged_transcript = get_postagged_transcript(transcript)
  unclean_para = predict_dial_tags(postagged_transcript)
  tran_para = remove_unwanted_spaces(unclean_para)

  # convert paragraph to summary
  words_num = len(tran_para.split())
  max_len = int(0.5 * words_num)
  min_len = int(0.25 * words_num)
  asummary = settings.ABS_SUMMARIZER(tran_para, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']

  return asummary