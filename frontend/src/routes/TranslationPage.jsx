// App.jsx
import { useRef, useState } from "react";
import WebcamSection from "../components/WebcamSection";
import TranslationSection from "../components/TranslationSection";
import AudioPlayer from "../components/AudioPlayer";
import axios from "axios";
import { showToast } from "../utils/showToast";
import { ToastContainer } from "react-toastify";

function TranslationPage() {
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [translatedText, setTranslatedText] = useState("");
  const [selectedLanguage, setSelectedLanguage] = useState("en-IN-rohan");
  const [audio, setaudio] = useState(null);
  const [isAudioGenerating, setisAudioGenerating] = useState(false);
  const [isTextGenterating, setisTextGenterating] = useState(false);
  const [audioDuration, setaudioDuration] = useState(0);
  const [frames, setFrames] = useState([]);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const captureIntervalRef = useRef(null);

  const languages = [
    { id: "en-IN-rohan", name: "English", flag: "🇮🇳" },
    { id: "hi-IN-shaan", name: "Hindi", flag: "🇮🇳" },
    { id: "bn-IN-ishani", name: "Bengali", flag: "🇮🇳" },
    { id: "es-MX-carlos", name: "Spanish", flag: "🇪🇸" },
    { id: "fr-FR-natalie", name: "French", flag: "🇫🇷" },
    { id: "de-DE-miles", name: "German", flag: "🇩🇪" },
    { id: "ja-JP-kenji", name: "Japanese", flag: "🇯🇵" },
  ];

  const handleStartSession = () => {
    setIsSessionActive(true);
    setTranslatedText("");
  };

  // Function to start capturing frames
  const startCapturingFrames = () => {
    if (!canvasRef.current) {
      canvasRef.current = document.createElement("canvas");
    }
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    canvas.width = 640;
    canvas.height = 480;

    captureIntervalRef.current = setInterval(() => {
      if (video && !video.paused && !video.ended) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const frameData = canvas.toDataURL("image/jpeg"); // base64 image
        setFrames((prevFrames) => [...prevFrames, frameData]); // push into state
      }
    }, 200); // capture every 200ms (~5 fps)
  };

  const handleStopSession = () => {
    setIsSessionActive(false);
    // setTranslatedText("");
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
  };

  const handleGenerateVoice = async () => {
    // This would connect to Murf API in a real application
    console.log("Generating voice for:", translatedText);
    console.log(selectedLanguage);

    try {
      setisAudioGenerating(true);
      const res = await axios.post(
        `${import.meta.env.VITE_MURF_URL}/v1/speech/generate`,
        {
          text: translatedText,
          voiceId: selectedLanguage || "en-IN-aarav",
          style: "Conversational",
        },
        {
          headers: {
            "Content-Type": "application/json",
            "api-key": `${import.meta.env.VITE_MURF_API_KEY}`,
          },
        }
      );

      console.log(res);
      setaudio(res.data.audioFile);
      setaudioDuration(res.data.audioLengthInSeconds);
    } catch (error) {
      console.log(error);
      showToast(
        error.response.data.message ||
          error.message ||
          "Faild to generate audio",
        1
      );
    } finally {
      setisAudioGenerating(false);
    }
  };

  const handleLanguageChange = (e) => {
    setSelectedLanguage(e.target.value);
  };

  const generateText = async () => {
    console.log(frames);
    const formData = new FormData();

    // append each frame to "files"
    frames.forEach((frame, index) => {
      const blob = dataURLtoBlob(frame); // convert base64 → Blob
      formData.append("files", blob, `frame_${index}.jpg`);
    });

    try {
      setisTextGenterating(true);
      // const response = await axios.post(
      //   `${import.meta.env.VITE_BACKEND_URL}/process-frames/`,
      //   formData,
      //   {
      //     headers: {
      //       "Content-Type": "multipart/form-data",
      //     },
      //   }
      // );

      // console.log("Response:", response.data);
      // setTranslatedText(response.data.text);

      await new Promise((res , rej)=>{
        setTimeout(() => {
          setTranslatedText("Hello nice to meet you what are you doing")
        }, 5000);
      })
    } catch (error) {
      showToast(
        error.response.data.message ||
          error.message ||
          "Error uploading frames",
        1
      );
      console.error("Error uploading frames:", error);
    } finally {
      setisTextGenterating(false);
    }

    // setTimeout(() => {
    //   setTranslatedText(
    //     "My school is a place of learning, growth, and creativity. It provides a positive environment where students not only gain knowledge from books but also develop essential life skills. The teachers are dedicated and supportive, always encouraging us to reach our full potential. Along with academics, my school also focuses on extracurricular activities like sports, cultural events, and competitions, which help us build confidence and teamwork. The classrooms are bright and welcoming, and the library and laboratories make learning more engaging. I feel proud to be a student of my school because it truly shapes us into responsible and capable individuals."
    //   );
    // }, 1000);
  };

  return (
    <>
      <ToastContainer />
      <div className="min-h-screen  text-gray-200">
        <main className="container mx-auto px-4 py-8 max-w-6xl space-y-8">
          {/* <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8"> */}
          <WebcamSection
            isSessionActive={isSessionActive}
            onStartSession={handleStartSession}
            onStopSession={handleStopSession}
            startCapturingFrames={startCapturingFrames}
            videoRef={videoRef}
            generateText={generateText}
            isTextGenterating={isTextGenterating}
            frames={frames}
          />

          <TranslationSection
            translatedText={translatedText}
            selectedLanguage={selectedLanguage}
            languages={languages}
            onGenerateVoice={handleGenerateVoice}
            onLanguageChange={handleLanguageChange}
            audioFileUrl={audio}
            isAudioGenerating={isAudioGenerating}
          />
          {/* </div> */}

          <AudioPlayer
            src={audio}
            language={selectedLanguage}
            audioDuration={audioDuration}
          />
          {/* <RecentTranslations translations={recentTranslations} /> */}
        </main>
      </div>
    </>
  );
}

export default TranslationPage;

function dataURLtoBlob(dataURL) {
  const arr = dataURL.split(",");
  const mime = arr[0].match(/:(.*?);/)[1];
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  while (n--) {
    u8arr[n] = bstr.charCodeAt(n);
  }
  return new Blob([u8arr], { type: mime });
}
