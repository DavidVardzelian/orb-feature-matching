# ORB Feature Matching Across Video Streams 🎥  

This project demonstrates **feature matching** between two video perspectives using **ORB (Oriented FAST and Rotated BRIEF)** and **FLANN-based matcher**. It visualizes keypoints and dynamically draws lines connecting matches across frames.  

## 📽️ Project Overview  
The project processes two videos frame by frame, detects keypoints, and draws matches, creating a visually appealing output video.  
**Resolution:** 1280x720 | **Frame Rate:** 60 FPS  

### 🔧 Tech Stack  
- **OpenCV** – Computer vision processing  
- **Python** – Primary language  
- **NumPy** – Matrix operations  
- **ORB (Oriented FAST and Rotated BRIEF)** – Feature detection  
- **FLANN** – Fast descriptor matching  

### 📄 How It Works  
1. **Video Input** – Two videos are processed in parallel.  
2. **Feature Detection** – ORB extracts 1000 keypoints per frame.  
3. **Feature Matching** – FLANN matcher finds the best matches between the two perspectives.  
4. **Visualization** – Matched keypoints are connected with dynamic colored lines across the frames.  
5. **Export** – The final video is rendered at 60 FPS with 1280x720 resolution, suitable for LinkedIn upload.  

### ▶️ How to Run  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/orb-feature-matching.git
   cd orb-feature-matching

