import React, { useState } from 'react';
import './css/upload.css'
import Form from 'react-bootstrap/Form';
import 'bootstrap/dist/css/bootstrap.min.css';

import './css/upload.css'
import { ThreeDots } from  'react-loader-spinner'
// import "react-loader-spinner/dist/loader/css/react-spinner-loader.css";


function Upload(props) {
  const setVisible = props.setVisible;
  const visible = props.visible;

  const [file, setFile] = useState();
  const [numspeakers, setNumspeakers] = useState(0);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [useSI, setUseSI] = useState(false);
  
  const setAsummary = props.setAsummary;
  const setEsummary = props.setEsummary;
  const setTranscript = props.setTranscript;

  const changeHandler = (event) => {
    setFile(event.target.files[0]);
	};

  const submitAudio = (e) => {
    e.preventDefault();
    setLoading(true);

    const formData = new FormData();
    formData.append('audio', file);
    formData.append('num_speakers', numspeakers);
    formData.append('use_si', useSI);
    
    const options = { 
      method: 'post',
      headers: {
        'Accept': 'application/json'
      },
      body: formData
    }

    fetch(`${process.env.REACT_APP_API_URL}/summariser/getsummary/`, options)
    .then(response => {
      if (response.ok) {
        return response.json()
      } else {
        return response.json().then(text => {throw text})
      }
    }).then(d => {
      setLoading(false)
      setVisible(false)
      alert(`Summary Generated !!`)
      document.getElementById('displaySummary').scrollIntoView({ behavior: 'smooth' });
      setError(null)
      setTranscript(d.transcript)
      setEsummary(d.esummary)
      setAsummary(d.asummary)
    }).catch(err => {
        setLoading(false)
        if (err.Error === "RuntimeError") {
          setError('Audio file is of invalid format !!')
        } else if (err.Message) {
          setError(err.Message);
        } else {
          setError('Please try submitting audio again...');
        }
    })
  }

    

  return (
      <div className='upload'>
        <h2>Upload Your File</h2>
        <img src={require('../assets/removed.png')} alt='bg' />
        

        <input className='input' type="file" name="file" accept="audio/wav" onChange={changeHandler}/>
        <label className='spkrinput'>
         Number of Speakers :
          <input size={1} type="text" name="numspeakers" onChange={(e) => setNumspeakers(e.target.value)}/>
        </label>
         {/* SPEAKER IDENTIFICATION SWITCH */}
         <label className='checkbox'>
          <Form>
            <Form.Check 
              type="switch"
              id="custom-switch"
              label="Speaker Identification"
              onClick={(e) => setUseSI(!useSI)}
            />
          </Form>
        </label>

        {loading === false && <button className='upbtn' type="button" onClick={submitAudio}>Submit</button>}
        {loading === true && <ThreeDots 
                                height="80" 
                                width="80" 
                                radius="9"
                                color="#E64830" 
                                ariaLabel="three-dots-loading"
                                wrapperStyle={{}}
                                wrapperClassName=""
                                visible={true}
                                />}
        {error && <p className='error'>{ error }</p>}
        <button onClick={() => setVisible(!visible)} className='close'>Close</button>
        
      </div>


  )
}

export default Upload