// components/AudioPlayer.jsx
import {
  Play,
  Pause,
  Volume2,
  VolumeX,
  Download,
  FileAudio,
} from "lucide-react";
import { useRef, useState, useEffect } from "react";

const AudioPlayer = ({
  src,
  title = "Audio File",
  language = "Unknown",
  audioDuration,
}) => {
  const audioRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(
    Number(audioDuration || 0).toFixed(2)
  );
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const updateTime = () => setCurrentTime(audio.currentTime);
    const updateDuration = () =>
      setDuration(Number(audioDuration || 0).toFixed(2));
    const handleEnd = () => setIsPlaying(false);

    audio.addEventListener("timeupdate", updateTime);
    audio.addEventListener("loadedmetadata", updateDuration);
    audio.addEventListener("ended", handleEnd);

    return () => {
      audio.removeEventListener("timeupdate", updateTime);
      audio.removeEventListener("loadedmetadata", updateDuration);
      audio.removeEventListener("ended", handleEnd);
    };
  }, [audioDuration]);

  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = volume;
      setIsMuted(volume === 0);
    }
  }, [volume]);

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleTimeChange = (newTime) => {
    setCurrentTime(newTime);
    if (audioRef.current) {
      audioRef.current.currentTime = newTime;
    }
  };

  const toggleMute = () => {
    if (!audioRef.current) return;
    if (isMuted) {
      setVolume(1);
    } else {
      setVolume(0);
    }
    setIsMuted(!isMuted);
  };

  const handleVolumeChange = (e) => {
    const newVolume = parseFloat(e.target.value);
    setVolume(newVolume);
    setIsMuted(newVolume === 0);
  };

  const downloadAudio = () => {
    if (!src) return;
    const link = document.createElement("a");
    link.href = src;
    link.download = `${title.replace(/\s+/g, "_")}.mp3`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  useEffect(() => {
    setIsPlaying(false);
  }, [src]);

  if (!src) return null;
  console.log(Number(audioDuration || 0).toFixed(2));

  return (
    <div className="w-full max-w-xl mx-auto mt-10 bg-gray-900 border border-gray-800 rounded-2xl p-6 shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <FileAudio className="w-6 h-6 text-blue-400" />
          <h2 className="text-lg font-semibold text-gray-100">{title}</h2>
        </div>
        <button
          onClick={downloadAudio}
          className="p-2 rounded-full cursor-pointer bg-gray-800 hover:bg-gray-700 transition-colors"
          title="Download audio"
        >
          <Download className="w-4 h-4 text-gray-300 " />
        </button>
      </div>

      <p className="text-sm text-gray-400 mb-4">Language: {language}</p>

      {/* Progress bar */}
      <ProgressBar
        currentTime={currentTime}
        duration={duration}
        onTimeChange={handleTimeChange}
        className="mb-5"
      />

      <div className="flex items-center justify-between">
        <button
          onClick={togglePlay}
          className="p-3 cursor-pointer rounded-full bg-blue-500 hover:bg-blue-600 transition-colors"
        >
          {isPlaying ? (
            <Pause className="w-5 h-5 text-white" />
          ) : (
            <Play className="w-5 h-5 text-white" />
          )}
        </button>

        <div className="flex items-center gap-3">
          <button
            onClick={toggleMute}
            className="text-gray-300 hover:text-white"
          >
            {isMuted || volume === 0 ? (
              <VolumeX className="w-5 h-5" />
            ) : (
              <Volume2 className="w-5 h-5" />
            )}
          </button>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={volume}
            onChange={handleVolumeChange}
            className="w-20 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
            style={{
              background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${
                volume * 100
              }%, #4b5563 ${volume * 100}%, #4b5563 100%)`,
            }}
          />
        </div>
      </div>

      <audio ref={audioRef} src={src} />
    </div>
  );
};

export default AudioPlayer;

const ProgressBar = ({
  currentTime,
  duration,
  onTimeChange,
  className = "",
}) => {
  const progressBarRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);

  const progressPercentage = duration > 0 ? (currentTime / duration) * 100 : 0;

  const handleProgressClick = (e) => {
    if (!progressBarRef.current) return;

    const progressBar = progressBarRef.current;
    const rect = progressBar.getBoundingClientRect();
    const clickPosition = e.clientX - rect.left;
    const percentage = clickPosition / rect.width;
    const newTime = percentage * duration;

    onTimeChange(newTime);
  };

  const handleDragStart = (e) => {
    setIsDragging(true);
    handleProgressClick(e);

    const handleDrag = (e) => {
      if (!isDragging) return;
      handleProgressClick(e);
    };

    const handleDragEnd = () => {
      setIsDragging(false);
      document.removeEventListener("mousemove", handleDrag);
      document.removeEventListener("mouseup", handleDragEnd);
      document.removeEventListener("touchmove", handleDrag);
      document.removeEventListener("touchend", handleDragEnd);
    };

    document.addEventListener("mousemove", handleDrag);
    document.addEventListener("mouseup", handleDragEnd);
    document.addEventListener("touchmove", handleDrag);
    document.addEventListener("touchend", handleDragEnd);
  };

  return (
    <div className={`w-full ${className}`}>
      {/* Time display above progress bar */}
      <div className="flex justify-between text-xs text-gray-400 mb-2">
        <span>{formatTime(currentTime)}</span>
        <span>{formatTime(duration)}</span>
      </div>

      <div
        ref={progressBarRef}
        className="w-full h-2 bg-gray-700 rounded-full cursor-pointer relative group"
        onClick={handleProgressClick}
      >
        {/* Background track */}
        <div className="absolute inset-0 bg-gray-600 rounded-full"></div>

        {/* Progress track */}
        <div
          className="absolute h-full bg-blue-500 rounded-full transition-all duration-300"
          style={{ width: `${progressPercentage}%` }}
        ></div>

        {/* Progress handle - Always visible and draggable */}
        <div
          className="absolute top-1/2 w-4 h-4 bg-white rounded-full shadow-lg transform -translate-y-1/2 -translate-x-1/2 cursor-pointer active:scale-110 transition-transform"
          style={{ left: `${progressPercentage}%` }}
          onMouseDown={handleDragStart}
          onTouchStart={handleDragStart}
        ></div>

        {/* Hover preview tooltip */}
        <div
          className="absolute bottom-full mb-2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none"
          style={{ left: `${progressPercentage}%` }}
        >
          <div className="bg-gray-800 text-white text-xs py-1 px-2 rounded">
            {formatTime(currentTime)}
          </div>
        </div>
      </div>
    </div>
  );
};

// Helper function to format time
const formatTime = (time) => {
  if (isNaN(time) || time === Infinity) return "0:00";
  const minutes = Math.floor(time / 60);
  const seconds = Math.floor(time % 60);
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
};
