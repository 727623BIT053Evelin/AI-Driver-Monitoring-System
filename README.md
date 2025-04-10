# 🚗 AI Driver Monitoring System  
*Real-time driver behavior analysis for enhanced road safety*  

---

## 🔍 Key Features  
### 🚨 Violation Detection  
- **Distractions**: Phone use 📱, Smoking 🚬, Eating 🍔, Alcohol 🍷  
- **Drowsiness**: Eye closure detection 😴 (EAR algorithm)  
- **Vehicle Safety**: Overcrowding alert in driver seat 👥  

### 🚔 Alert System  
- 🔊 Audio warnings  
- 🗣️ Text-to-speech alerts  
- 📱 Telegram notifications with snapshots + fine amount  

## Telegram Notification
![Screenshot 2025-04-10 163047](https://github.com/user-attachments/assets/9a864f0b-3aa1-4398-9b29-983aeb848725)


### 🌐 Web Dashboard  
## Dashboard
![Screenshot 2025-04-10 163155](https://github.com/user-attachments/assets/869644a8-31f9-4a05-a638-c442961adee3)

- Real-time camera feed 🎥  
- Violation statistics 📈  
- One-click controls 🕹️  

---

## ⚙️ Tech Stack  
**Computer Vision**:  
- OpenCV  
- YOLOv8  
- Dlib (facial landmarks)  

**Backend**:  
- Python 3.8+  
- Node.js (web interface)  

**Alerts**:  
- Pygame (audio)  
- Telegram Bot API  
- pyttsx3 (TTS)  

---

## 🛠️ Installation  
```bash
# Clone repository
git clone https://github.com/yourusername/driver-monitoring-system.git
cd driver-monitoring-system

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies (for web dashboard)
npm install


🚀 Usage
Configure Telegram (optional):

python
# In AI_DRIVER_MONITORING.py
TELEGRAM_BOT_TOKEN = 'your_bot_token_here'
TELEGRAM_CHAT_ID = 'your_chat_id_here'
Start the system:

bash
# Start Python detection
python AI_DRIVER_MONITORING.py

# Start web dashboard (separate terminal)
node server.js
Access dashboard at http://localhost:5000
