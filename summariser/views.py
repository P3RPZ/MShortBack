from rest_framework import status, generics, parsers
from rest_framework.response import Response
from drf_yasg.utils import swagger_auto_schema

from .models import Audio
from .serializers import AudioInputSerializer, SummaryOutputSerializer
from django.conf import settings
import ffmpeg
import speech_recognition as sr
import os
import shutil
from .utils import (
  get_aaitranscript, 
  assemblyai_transcript, 
  get_speakeridentity, 
  preprocess_transcript, 
  get_esummary, 
  esumm_to_str,
  get_asummary
)

import time

class GenerateSummary(generics.GenericAPIView):
    serializer_class = AudioInputSerializer
    parser_classes = (parsers.FormParser, parsers.MultiPartParser, parsers.FileUploadParser)

    @swagger_auto_schema(operation_description="Summary Generation (Transcript)",
                         responses={ 201: 'Generated Successfully',
                                400: 'Given audio is of invalid format'})

    def post(self, request):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data

        try:
            audio = Audio(audio=data['audio'], num_speakers=data['num_speakers'], use_si=data['use_si'])
            audio.save()

            start_time = time.time()

            if (audio.num_speakers is not None and audio.num_speakers != 0):
              diarization = settings.PIPELINE(audio.audio.path, num_speakers=audio.num_speakers)
            else:
              diarization = settings.PIPELINE(audio.audio.path)
            diarize_results = list()
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
                if (round(turn.start, 1) < round(turn.end, 1)):
                  diarize_results.append([speaker, round(turn.start, 1), round(turn.end, 1)])
            print("Diarisation Completed !!")            
            print(diarize_results)

            #finding largest duration cut for each speaker for identification
            largest_duration = {}

            for i, data in enumerate(diarize_results):
              speaker_id = data[0]
              duration = data[2] - data[1]
              
              # If this is the first time we're seeing this speaker, add their dictionary to the largest_duration dictionary
              if speaker_id not in largest_duration:
                  largest_duration[speaker_id] = {"index": i, "duration": duration}
              
              # If we've seen this speaker before, compare the current duration to their previous largest duration
              else:
                  if duration > largest_duration[speaker_id]["duration"]:
                      largest_duration[speaker_id]["index"] = i
                      largest_duration[speaker_id]["duration"] = duration
              
            # print(largest_duration)

            r = sr.Recognizer()
            audio_input = ffmpeg.input(audio.audio.path)

            name = os.path.basename(audio.audio.name)

            if not os.path.exists('audios/cuts/'):
              os.mkdir('audios/cuts/')
            
            for i, d in enumerate(diarize_results):
                audio_cut = audio_input.audio.filter('atrim', start=d[1], end=d[2])
                audio_output = ffmpeg.output(audio_cut, f'audios/cuts/{name}_cut{i+1}.wav')
                ffmpeg.run(audio_output)
            
            si_results = {}
            if audio.use_si:
              for key, value in largest_duration.items():
                speaker = get_speakeridentity(f"audios/cuts/{name}_cut{value['index']+1}.wav")
                if speaker not in si_results.values():
                  si_results[key] = speaker
                else:
                  si_results[key] = key
              print("Speaker Identification Completed")
              print(si_results)
            else:
              for key in largest_duration.keys():
                si_results[key] = key

            t_ids = list()
            transcript = list()

            for i, d in enumerate(diarize_results):

                with sr.AudioFile(f'audios/cuts/{name}_cut{i+1}.wav') as source:
                    # r.adjust_for_ambient_noise(source)
                    aud = r.record(source)
                
                id = assemblyai_transcript(f'audios/cuts/{name}_cut{i+1}.wav')
                t_ids.append(id)
            
            prev_speaker = ""
            # speakerwise_trans = dict()
            for id, d in zip(t_ids, diarize_results):
                try:
                  tran = get_aaitranscript(id)
                  if tran != "":
                    if prev_speaker != d[0]:
                      transcript.append(f"{si_results[d[0]]}: {tran}")
                      # if d[0] not in speakerwise_trans:
                      #   speakerwise_trans[si_results[d[0]]] = tran
                      # else:
                      #   speakerwise_trans[si_results[d[0]]] += tran
                      prev_speaker = d[0]
                    else:
                      prev_trans = transcript.pop()
                      prev_trans += f" {tran}"
                      transcript.append(prev_trans)
                      # speakerwise_trans[si_results[d[0]]] += tran
                except:
                  # transcript.append(f"{d[0]}: ERROR OCCURRED !!!!")
                  print(f"{si_results[d[0]]}: ERROR OCCURRED !!!!")
            print("Transcript Generated")

            shutil.rmtree('audios/cuts/')
            os.remove(audio.audio.path)

            str_transcript = '\n\n'.join(transcript)
            speakerwise_trans = preprocess_transcript(str_transcript)
            esummary = get_esummary(speakerwise_trans)
            print("Extractive Summary Generated")

            asummary = get_asummary(transcript)
            print("Abstractive Summary Generated")
            
            audio.transcript = str_transcript
            audio.e_summary = esumm_to_str(esummary)
            audio.a_summary = asummary
            audio.save()

            print(f"Total time taken = {round(time.time() - start_time, 2)}")
            return Response({'Success': "Summary Generated Successfully", 'transcript': str_transcript, 'esummary': esummary, 'asummary': asummary}, status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response({ "Error": type(e).__name__ , "Message": str(e)}, status=status.HTTP_409_CONFLICT)
        else:
            print(serializer.errors)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


