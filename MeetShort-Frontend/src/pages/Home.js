
import React, {useState } from 'react';
import '../App.css'
import HomePage from './HomePage'
import About from './About'
import How from './How'


import './css/home.css'
import Display from './Display'



function Home() {
  const [transcript, setTranscript] = useState();
  const [esummary, setEsummary] = useState({});
  const [asummary, setAsummary] = useState();

  return (
    <div className='App'>
    <HomePage setAsummary={setAsummary} setEsummary={setEsummary} setTranscript={setTranscript}/>
    <div className='secnd'>
      <About/>
      <How/>
    </div>
    <Display asummary={asummary} esummary={esummary} transcript={transcript}/>
  </div>
  )
}

export default Home