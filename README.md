# Smart Irrigation System

This project utilizes sensor data to predict which parcel(s) of land require irrigation. It leverages machine learning techniques to identify patterns in the environment and activate irrigation pumps accordingly.

---

### 📁 Files Included

- **Dataset.csv**: Sensor readings and irrigation labels for three land parcels.
- **Irrigation_Model_Train.ipynb**: Python notebook to preprocess data, train, and evaluate a multi-output classifier.
- **Model_Deployment_Streamlit.py**: Streamlit app for deploying the trained model interactively.
- **Farm_Irrigation_System.pkl**: Trained model saved using `joblib`.

---

### 🧠 ML Approach

- Preprocessing with `MinMaxScaler`
- Random Forest wrapped in `MultiOutputClassifier` to support multi-label classification.
- Evaluation using `classification_report`

---

### 💻 Deployment

To launch the Streamlit app:

```bash
streamlit run Model_Deployment_Streamlit.py
