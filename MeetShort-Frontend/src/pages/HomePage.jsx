import React, { useState } from 'react';
import NavigationBar from './NavigationBar'
import Upload from './Upload';
import './css/homepage.css'


function HomePage(props) {
  const setAsummary = props.setAsummary;
  const setEsummary = props.setEsummary;
  const setTranscript = props.setTranscript;

  const [visible, setVisible] = useState(true);
  return (
    <div className='homepage'>
      <NavigationBar/>
        <div className="meetshort">
            <h1>Meeting Summarizer</h1>
            <p>Summarize you meetings in one click. Just provide the audio of your meeting and number of speakers (optional). Then wait and watch the magic. You can get transcript and 2 types of summaries from the audio.</p>
            <button className='btn-start' onClick={() => setVisible(!visible)}>Get Started<i class='fas fa-angle-double-right'></i></button>
        </div>
        {!visible && <Upload setVisible={setVisible} visible={visible} setAsummary={setAsummary} setEsummary={setEsummary} setTranscript={setTranscript}/>}
        
    </div>
  )
}

export default HomePage