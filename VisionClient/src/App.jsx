import React, { useEffect, useRef, useState } from 'react';
import './App.css';

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [alertMsg, setAlertMsg] = useState('');
  const [synth] = useState(window.speechSynthesis);
  const [voices, setVoices] = useState([]);

  useEffect(() => {
    const loadVoices = () => setVoices(synth.getVoices());
    synth.onvoiceschanged = loadVoices;
    loadVoices();
  }, [synth]);

  useEffect(() => {
    const getCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error('Error accessing camera:', err);
      }
    };
    getCamera();
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      if (videoRef.current && canvasRef.current) {
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(async (blob) => {
          const formData = new FormData();
          formData.append('frame', blob, 'frame.jpg');
          try {
            const response = await fetch(`${import.meta.env.VITE_API_URL}/process_frame`, {
              method: 'POST',
              body: formData,
            });
            const data = await response.json();
            if (data.alert && data.message !== alertMsg) {
              setAlertMsg(data.message);
              const utter = new SpeechSynthesisUtterance(data.message);
              if (voices.length) utter.voice = voices[0];
              synth.speak(utter);
            } else if (!data.alert) {
              setAlertMsg('');
            }
          } catch (err) {
            console.error('Error sending frame:', err);
          }
        }, 'image/jpeg');
      }
    }, 1000);
    return () => clearInterval(interval);
  }, [alertMsg, voices, synth]);

  return (
    <div className="container">
      <div className="header-container">
        <h1 className="vision-text">Vision</h1>
        <h1 className="assistant-text">Aid Assistant</h1>
      </div>
      <h2 className="subtitle">Awareness in Every Frame</h2>
      <div>
        <video ref={videoRef} autoPlay muted className="video-feed" />
        <canvas ref={canvasRef} style={{ display: 'none' }} />
      </div>
      {alertMsg && <div className="alert">{alertMsg}</div>}
    </div>
  );
}