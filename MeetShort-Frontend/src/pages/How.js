import React from 'react'
import './css/how.css'

function how() {
  return (
    <div className='how'>
      <h2>How Does MeetShort Work?</h2>
      <div className='howcontent'>
      <p>First step is the Speaker Diarisation. Speaker diarisation
        is the process of segmenting an audio or video recording into segments or clusters,
        based on who is speaking. In other words, it is the process of separating out the
        different speakers in a recording and identifying which segments of the recording
        correspond to which speaker.</p>
        <p>Next step is Automatic Speech Recognition (ASR). Its the process by which a computer system can recognize spoken language and convert it into text or another machine-readable format. It generates the transcript that is displayed as result.</p>
        <p>Final step is Text Summarisation. Text summarization is the process of creating a shorter version of a longer text while preserving the most important information. There are two main approaches to text summarization: extractive and abstractive. Extractive summarization involves selecting a subset of the original text that best represents the most important information. Abstractive summarization, on the other hand, involves generating a summary that is not a direct subset of the original text.</p>
      </div>
    </div>
  )
}

export default how