import { useEffect, useState } from "react";
import "./App.css";

const API_URL = "https://cricket-shot-classification.onrender.com/analyze";

const MODEL_STATS = [
  {
    value: "69.60%",
    label: "Test Accuracy",
  },
  {
    value: "10",
    label: "Shot Classes",
  },
  {
    value: "16",
    label: "Frames / Video",
  },
];

function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [previewError, setPreviewError] = useState(false);

  useEffect(() => {
    return () => {
      if (videoUrl) {
        URL.revokeObjectURL(videoUrl);
      }
    };
  }, [videoUrl]);

  const handleVideoChange = (event) => {
    const file = event.target.files[0];

    setResult(null);
    setError("");
    setPreviewError(false);

    if (videoUrl) {
      URL.revokeObjectURL(videoUrl);
    }

    if (!file) {
      setVideoFile(null);
      setVideoUrl("");
      return;
    }

    const allowedExtensions = [
      ".avi",
      ".mp4",
      ".mov",
      ".mkv",
    ];

    const fileName = file.name.toLowerCase();

    const isSupported = allowedExtensions.some(
      (extension) => fileName.endsWith(extension)
    );

    if (!isSupported) {
      setError(
        "Unsupported video format. Please use AVI, MP4, MOV or MKV."
      );

      setVideoFile(null);
      setVideoUrl("");

      return;
    }

    setVideoFile(file);

    const previewUrl = URL.createObjectURL(file);

    setVideoUrl(previewUrl);
  };

  const handleAnalyze = async () => {
    if (!videoFile) {
      setError(
        "Please select a cricket video first."
      );
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();

    formData.append(
      "video",
      videoFile
    );

    try {
      const response = await fetch(
        API_URL,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.error ||
            "Video analysis failed."
        );
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message ||
          "Unable to connect to the backend."
      );
    } finally {
      setLoading(false);
    }
  };

  const formatShotName = (name) => {
    if (!name) {
      return "";
    }

    return name
      .split("_")
      .map(
        (word) =>
          word.charAt(0).toUpperCase() +
          word.slice(1)
      )
      .join(" ");
  };

  const getConfidenceLabel = () => {
    if (!result) {
      return "";
    }

    return `${Number(
      result.confidence_percent
    ).toFixed(2)}%`;
  };

  return (
    <div className="app">
      <div className="background-orb background-orb-one" />
      <div className="background-orb background-orb-two" />

      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <div className="brand-mark">
              CV
            </div>

            <div>
              <strong>
                CricketVision
              </strong>

              <span>
                Deep Learning Analytics
              </span>
            </div>
          </div>

          <div className="system-status">
            <span className="status-dot" />
            System Ready
          </div>
        </div>
      </header>

      <main className="page-container">
        <section className="hero">
          <div className="hero-copy">
            <div className="hero-badge">
              AI-Powered Cricket Analysis
            </div>

            <h1>
              Understand every
              <span> cricket shot</span>
            </h1>

            <p>
              Upload a batting video and analyze it using
              pretrained MobileNetV2 visual features, GRU-based
              temporal classification, and MediaPipe pose analytics.
            </p>

            <div className="hero-stats">
              {MODEL_STATS.map(
                (item) => (
                  <div
                    className="hero-stat"
                    key={item.label}
                  >
                    <strong>
                      {item.value}
                    </strong>

                    <span>
                      {item.label}
                    </span>
                  </div>
                )
              )}
            </div>
          </div>

          <div className="hero-model-card">
            <div className="model-card-top">
              <div>
                <span className="mini-label">
                  Final Architecture
                </span>

                <h3>
                  MobileNetV2 + GRU
                </h3>
              </div>

              <div className="model-icon">
                AI
              </div>
            </div>

            <div className="architecture-flow">
              <span>
                Video
              </span>

              <i>→</i>

              <span>
                MobileNetV2
              </span>

              <i>→</i>

              <span>
                GRU
              </span>

              <i>→</i>

              <span>
                Prediction
              </span>
            </div>

            <div className="model-meta">
              <div>
                <span>
                  Spatial Model
                </span>

                <strong>
                  MobileNetV2
                </strong>
              </div>

              <div>
                <span>
                  Temporal Model
                </span>

                <strong>
                  GRU
                </strong>
              </div>

              <div>
                <span>
                  Pose Engine
                </span>

                <strong>
                  MediaPipe
                </strong>
              </div>
            </div>
          </div>
        </section>

        <section className="workspace-grid">
          <div className="panel upload-panel">
            <div className="panel-heading">
              <div>
                <span className="section-kicker">
                  Video Input
                </span>

                <h2>
                  Upload Cricket Video
                </h2>

                <p>
                  Select a short batting clip for classification
                  and pose analysis.
                </p>
              </div>

              <div className="step-indicator">
                Step 01
              </div>
            </div>

            <label className="file-input">
              <div className="upload-icon">
                ↑
              </div>

              <div className="upload-text">
                <strong>
                  {videoFile
                    ? videoFile.name
                    : "Choose cricket video"}
                </strong>

                <span>
                  AVI, MP4, MOV or MKV
                </span>
              </div>

              <div className="browse-chip">
                Browse
              </div>

              <input
                type="file"
                accept=".avi,.mp4,.mov,.mkv,video/*"
                onChange={handleVideoChange}
              />
            </label>

            {videoUrl && !previewError && (
              <div className="video-wrapper">
                <video
                  src={videoUrl}
                  controls
                  className="video-preview"
                  onError={() =>
                    setPreviewError(true)
                  }
                />
              </div>
            )}

            {previewError && videoFile && (
              <div className="preview-message">
                <div className="preview-icon">
                  ▶
                </div>

                <div>
                  <strong>
                    Browser preview unavailable
                  </strong>

                  <p>
                    This format can still be analyzed normally
                    by the backend.
                  </p>
                </div>
              </div>
            )}

            <button
              className="analyze-button"
              onClick={handleAnalyze}
              disabled={
                !videoFile || loading
              }
            >
              {loading ? (
                <>
                  <span className="spinner" />
                  Analyzing Video...
                </>
              ) : (
                <>
                  <span>
                    ✦
                  </span>
                  Analyze Cricket Shot
                </>
              )}
            </button>

            {error && (
              <div className="error-message">
                {error}
              </div>
            )}
          </div>

          <aside className="panel info-panel">
            <span className="section-kicker">
              Analysis Pipeline
            </span>

            <h2>
              What happens next?
            </h2>

            <div className="pipeline-list">
              <div className="pipeline-item">
                <span className="pipeline-number">
                  01
                </span>

                <div>
                  <strong>
                    Frame Sampling
                  </strong>

                  <p>
                    The video is converted into 16 normalized frames.
                  </p>
                </div>
              </div>

              <div className="pipeline-item">
                <span className="pipeline-number">
                  02
                </span>

                <div>
                  <strong>
                    Feature Extraction
                  </strong>

                  <p>
                    MobileNetV2 extracts pretrained spatial features.
                  </p>
                </div>
              </div>

              <div className="pipeline-item">
                <span className="pipeline-number">
                  03
                </span>

                <div>
                  <strong>
                    Temporal Classification
                  </strong>

                  <p>
                    GRU analyzes the feature sequence and predicts the shot.
                  </p>
                </div>
              </div>

              <div className="pipeline-item">
                <span className="pipeline-number">
                  04
                </span>

                <div>
                  <strong>
                    Pose Analytics
                  </strong>

                  <p>
                    MediaPipe measures player pose detection across frames.
                  </p>
                </div>
              </div>
            </div>

            <div className="supported-shots">
              <span>
                Supported Classes
              </span>

              <div className="shot-tags">
                {[
                  "Cover",
                  "Defense",
                  "Flick",
                  "Hook",
                  "Late Cut",
                  "Lofted",
                  "Pull",
                  "Square Cut",
                  "Straight",
                  "Sweep",
                ].map(
                  (shot) => (
                    <small key={shot}>
                      {shot}
                    </small>
                  )
                )}
              </div>
            </div>
          </aside>
        </section>

        {result && (
          <section className="results-section">
            <div className="results-title-row">
              <div>
                <span className="section-kicker">
                  AI Analysis Complete
                </span>

                <h2>
                  Prediction Dashboard
                </h2>
              </div>

              <div
                className={`confidence-badge ${result.confidence_level}`}
              >
                {result.confidence_level}
                {" "}
                confidence
              </div>
            </div>

            <div className="prediction-hero">
              <div className="prediction-main">
                <span>
                  Predicted Shot
                </span>

                <h3>
                  {formatShotName(
                    result.prediction
                  )}
                </h3>

                <p>
                  The model assigns this classification
                  with a confidence score of
                  {" "}
                  <strong>
                    {getConfidenceLabel()}
                  </strong>.
                </p>
              </div>

              <div className="confidence-ring">
                <div className="confidence-ring-inner">
                  <strong>
                    {getConfidenceLabel()}
                  </strong>

                  <span>
                    Confidence
                  </span>
                </div>
              </div>
            </div>

            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-icon">
                  ◎
                </div>

                <span>
                  Confidence
                </span>

                <strong>
                  {Number(
                    result.confidence_percent
                  ).toFixed(2)}
                  %
                </strong>
              </div>

              <div className="metric-card">
                <div className="metric-icon">
                  ◉
                </div>

                <span>
                  Pose Detection
                </span>

                <strong>
                  {Number(
                    result.pose
                      .pose_detection_percent
                  ).toFixed(2)}
                  %
                </strong>
              </div>

              <div className="metric-card">
                <div className="metric-icon">
                  ▦
                </div>

                <span>
                  Frames Analyzed
                </span>

                <strong>
                  {
                    result.pose
                      .total_frames
                  }
                </strong>
              </div>

              <div className="metric-card">
                <div className="metric-icon">
                  ◇
                </div>

                <span>
                  Pose Frames
                </span>

                <strong>
                  {
                    result.pose
                      .frames_with_pose
                  }
                </strong>
              </div>
            </div>

            <div className="probability-card">
              <div className="probability-heading">
                <div>
                  <span className="section-kicker">
                    Probability Distribution
                  </span>

                  <h3>
                    Shot Classification Scores
                  </h3>
                </div>

                <span className="probability-note">
                  All 10 classes
                </span>
              </div>

              <div className="probability-list">
                {Object.entries(
                  result.probabilities
                )
                  .sort(
                    ([, a], [, b]) =>
                      b - a
                  )
                  .map(
                    ([
                      className,
                      probability,
                    ], index) => (
                      <div
                        className={`probability-row ${
                          index === 0
                            ? "probability-row-primary"
                            : ""
                        }`}
                        key={className}
                      >
                        <div className="probability-label">
                          <div>
                            <span className="rank">
                              {String(
                                index + 1
                              ).padStart(
                                2,
                                "0"
                              )}
                            </span>

                            <strong>
                              {formatShotName(
                                className
                              )}
                            </strong>
                          </div>

                          <span>
                            {(
                              Number(
                                probability
                              ) * 100
                            ).toFixed(2)}
                            %
                          </span>
                        </div>

                        <div className="progress-track">
                          <div
                            className="progress-bar"
                            style={{
                              width: `${
                                Number(
                                  probability
                                ) * 100
                              }%`,
                            }}
                          />
                        </div>
                      </div>
                    )
                  )}
              </div>
            </div>
          </section>
        )}

        <footer className="footer">
          <span>
            Cricket Shot Classification & Analytics
          </span>

          <span>
            MobileNetV2 • GRU • MediaPipe
          </span>
        </footer>
      </main>
    </div>
  );
}

export default App;