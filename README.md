# 🧠 AI Disease Prediction System (Logistic Regression)

Dual-disease demo app that predicts **Diabetes** and **Heart Disease** using Logistic Regression models.
This project contains generated placeholder datasets and trained models so you can run the app immediately.

## Project Structure
```
disease_prediction/
├── dataset/
│   ├── diabetes.csv
│   └── heart.csv
├── models/
│   ├── diabetes_model.pkl
│   └── heart_model.pkl
├── app.py
├── train_models.py
├── requirements.txt
└── README.md
```

## How to run
1. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   venv\Scripts\activate   # Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Notes
- The datasets in `dataset/` are synthetic placeholder data for demo and development purposes.
- You can replace the CSVs with real datasets (Kaggle Pima diabetes & UCI heart disease) and re-run `python train_models.py` to retrain models.
- Models were trained using Logistic Regression to be lightweight and interpretable.

## Model accuracies (on synthetic test splits)
- Diabetes model accuracy: 0.48
- Heart model accuracy: 0.56

## License
MIT