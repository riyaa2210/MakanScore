🏠 MakanScore – House Price Prediction
MakanScore is a machine learning + deep learning based web app that predicts house prices in India using a dataset of property features.  
It has a FastAPI backend (for prediction) and a React frontend (for UI).  
 🚀 Features
> Predicts house prices based on features like bedrooms, bathrooms, area, year, etc.
> FastAPI backend for handling ML model inference.
> React frontend with clean UI.
> Neural Network trained on Kaggle dataset.
> Preprocessing pipeline (scalers + encoders) saved for reproducibility.
📸 Screenshots
<img width="917" height="647" alt="image" src="https://github.com/user-attachments/assets/98a21024-767d-4d4c-8816-c9869eba3868" />

 ⚙️ Tech Stack
Frontend:React + Tailwind
Backend:FastAPI
ML/DL:TensorFlow, Scikit-learn

 
 >>Run Locally

Clone the repository:

git clone https://github.com/riyaa2210/MakanScore.git
cd MakanScore


>> Backend


cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 5000

 >>Frontend


cd frontend
npm install
npm start


 📌 Future Improvements

> Deploy on AWS / Render
> Add user authentication
> Compare different ML models


MakanScore/
├── backend/
├── frontend/
├── assets/
│    ├── ui-home.png
│    ├── ui-prediction.png
├── README.md

 
 3. Commit & Push

git add README.md assets/
git commit -m "Added README with UI screenshots"
git push origin main   # or master

