# NeuroScan AI – Brain Tumor Detection (MobileNetV2)

```
brain-tumor-app/
├── model/      train.py              <- train on Kaggle/Colab (GPU)
├── backend/    app.py (FastAPI)      <- serves the model as an API
└── frontend/   index.html/css/js     <- website UI
```

## Step 1 – Train the model (Kaggle or Colab, GPU on)
1. Upload `model/train.py` (Kaggle: add the dataset via "Add Input").
2. Run it. Download the two outputs:
   - `brain_tumor_mobilenetv2.keras`
   - `class_names.json`
3. Put both files inside the `backend/` folder.

## Step 2 – Run the backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```
Check http://127.0.0.1:8000/docs – you can test `/predict` there.

## Step 3 – Run the frontend
```bash
cd frontend
python -m http.server 5500
```
Open http://localhost:5500

(If your backend is on another URL, edit `API_URL` at the top of `frontend/script.js`.)

## API
`POST /predict` (multipart, field `file`) returns:
```json
{"prediction":"glioma","label":"Glioma Tumor","confidence":98.7,"has_tumor":true,
 "probabilities":{"Glioma Tumor":98.7,"Meningioma Tumor":0.8,"No Tumor Detected":0.3,"Pituitary Tumor":0.2}}
```

## Deployment ideas
- Backend: Render / Railway / Hugging Face Spaces (Docker)
- Frontend: Netlify / Vercel / GitHub Pages (update `API_URL`)

> Educational project only – not a medical diagnostic tool.

## Grad-CAM tumor localization
`/predict` also returns `gradcam` (a PNG data-URL with the heatmap blended on the MRI and a white box around the hottest region) and `location` (e.g. "upper-right region of the scan"). The website shows it under the results with an Overlay / Side-by-Side toggle. No retraining is needed – it works with the same model file.
