import React, { useState } from 'react'
import './css/display.css'

function Display(props) {
  const transcript = props.transcript;
  const esummary = props.esummary;
  const asummary = props.asummary;

  const [activebtn, setActivebtn] = useState('transcript')

  const changeBtn = (e) => {
    setActivebtn(e.target.name);
  }

  return (
    
    <div className='display' id='displaySummary'>
        <h3>Result</h3>
        {/* SUMMARY CHOSE BUTTONS */}
            <button className={ activebtn === 'transcript' ? 'summary-button active' : 'summary-button' } name='transcript' onClick={changeBtn}>Transcript</button>
            <button className={ activebtn === 'esummary' ? 'summary-button active' : 'summary-button' } name='esummary' onClick={changeBtn}>Extractive Summary</button>
            <button className={ activebtn === 'asummary' ? 'summary-button active' : 'summary-button' } name='asummary' onClick={changeBtn}>Abstractive Summary</button>

        <div className='summary-toggle'>
          <div className='asr'>
            {/* based on selection display summary */}
                {activebtn === 'transcript' ? 
                  <p className='transcript'>{transcript}</p> : 
                  activebtn === 'esummary' ? (
                  Object.entries(esummary).map(([key, value]) => (
                    <div>
                      <h4>{key}</h4>
                      <ul>
                        {value.map(item => (
                          <li>{item}</li>
                        ))}
                      </ul>
                    </div>
                  ))) : (
                    <p className='transcript'>{asummary}</p>
                  )}
                
          </div>
        </div>
    </div>
  )
}

export default Display