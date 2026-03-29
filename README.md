# 👁️ Eye-Blink-to-Speech: AI-Driven Assistive Communication

A sophisticated computer vision and signal processing system engineered for individuals with profound motor neuron diseases (like ALS or advanced-stage paralysis). It enables users to "speak" pre-defined phrases by performing intentional, duration-specific eye blinks, bridging the gap for non-invasive assistive communication.

![Project Demo](YOUR_GIF_URL)

---

## 🚀 Key Technological Innovations (The Data Science Pitch)
Recruiters: This project goes beyond simple object detection. It includes advanced implementations of the following concepts:

* **Real-time 468 Landmark Tracking (MediaPipe):** Uses **MediaPipe Face Mesh** to track facial structure with high fidelity (468 landmarks), significantly outperforming Haar Cascade or Dlib-based methods.
* **Signal Smoothing (Moving Average):** To filter out noisy data from natural head movement and lighting, the EAR data is passed through a rolling `deque` buffer, calculating a dynamic moving average for stable trigger detection.
* **Asynchronous Text-to-Speech (daemon threading):** I designed a dedicated **daemon thread** for the `AsyncTTS` class. This ensures the Text-to-Speech process does not block the main video loop, allowing the camera to maintain a steady **60 FPS** for accurate signal sampling.
* **Formally Modeled State Machine:** The detector uses a `DetectorState` enumeration and structured logic to manage transitions between **UNINITIALIZED**, **CALIBRATING**, and **ACTIVE** modes. This makes the system robust against state-based errors.

---

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Computer Vision:** OpenCV, MediaPipe (Face Mesh)
* **Signal Processing:** NumPy, Collections (rolling buffers)
* **Speech Synthesis:** Pyttsx3 (multi-threaded)

---

## ⚙️ Installation & Usage
1.  Clone the repository: `git clone https://github.com/Kunal2005jan15/EyeBlinkToSpeech.git`
2.  Navigate to the directory: `cd EyeBlinkToSpeech`
3.  Install dependencies: `pip install -r requirements.txt`
4.  Run the application: `python app.py`

---

## ⚖️ System Evaluation

### ✅ Advantages
* **Auto-Calibration:** Dynamically adjusts to the user’s eye structure and lighting, making it usable without complex manual tuning.
* **High Efficiency:** Leveraging TensorFlow Lite's XNNPACK delegate, the MediaPipe model runs extremely fast on standard CPU hardware.
* **Live HUD:** The real-time EAR graph provides visual confirmation of the system's threshold, enhancing user trust.

### ⚠️ Limitations
* **Menu Selection (Current V2.0.0):** It currently functions as a sequential menu, requiring the user to wait for the system to cycle through phrases. It does not yet have a direct selection matrix.
* **Lighting Dependence:** Extreme lighting changes can affect landmark accuracy.

---

## 🔮 Future Scope (Version 3.0.0)
I plan to implement the following features to advance the system further:

1.  **Selection Matrix with Morphemes:** Instead of cycling words sequentially, V3.0.0 will feature a predictive matrix. A patient can blink to choose **"Help"** then **"Pain"** then **"Medication,"** making the system context-aware.
2.  **Adaptive Auto-Thresholding:** I am researching techniques to make the threshold `(THR)` adjust on the fly based on the patient's fatigue level throughout the day.
3.  **Local Deep Learning Integration:** I am exploring fine-tuning a small, local LSTM model to understand a patient's unique long-blink pattern better, making it more personal and accurate.

---

## 👨‍💻 Author & Contributions
Project created by [Your Name]. This project is open to contributions. Feel free to submit an issue or pull request.

---

## 📜 Acknowledgments
Special thanks to the MediaPipe and OpenCV communities for their incredible documentation.