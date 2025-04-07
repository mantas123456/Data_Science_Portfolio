import os

# Define the root directory (adjust if you're running this from somewhere else)
ROOT_DIR = "."

# Define the structure
structure = {
    "Project_1_Sports_Prediction": {
        "subfolders": ["data", "notebooks", "models"],
        "readme": "# 🧠 Predictive Modeling of Sports Outcomes\n\n## Overview\nThis project builds machine learning models to predict the outcome of sports matches based on historical data.\n\n## Goals\n- Collect or simulate realistic sports datasets\n- Perform EDA and feature engineering\n- Train classification models (e.g., Logistic Regression, Random Forest, XGBoost)\n- Evaluate model performance (Accuracy, F1, ROC AUC)\n- Explore model interpretability (SHAP, feature importance)\n\n## Structure\n- `data/`: raw and processed datasets\n- `notebooks/`: Jupyter Notebooks for analysis and modeling\n- `models/`: saved models and pipelines\n\n## Results\nHighlights of best model performance and insights.\n\n## Future Work\n- Add ensemble techniques\n- Incorporate more contextual features (e.g., player form, injuries)"
    },
    "Project_2_Customer_Segmentation": {
        "subfolders": ["data", "scripts"],
        "readme": "# 🎯 Customer Segmentation in iGaming\n\n## Overview\nThis project segments players based on behavior, frequency, and value using unsupervised learning.\n\n## Goals\n- Clean and preprocess user activity data\n- Explore behavioral metrics (RFM analysis)\n- Apply KMeans or DBSCAN clustering\n- Visualize clusters and generate BI-style summaries\n\n## Structure\n- `data/`: player data (simulated or anonymized)\n- `scripts/`: analysis scripts and clustering pipeline\n\n## Results\nDescription of customer segments and actionable insights.\n\n## Future Work\n- Test different distance metrics\n- Integrate feedback loops or cluster evolution over time"
    },
    "Project_3_Model_Deployment": {
        "subfolders": ["app"],
        "readme": "# 🚀 Deploying ML Models as APIs\n\n## Overview\nThis project demonstrates how to deploy an ML model using a lightweight REST API.\n\n## Goals\n- Train a simple classification/regression model\n- Serialize it using joblib/pickle\n- Build a REST API using Flask or FastAPI\n- Test locally and optionally containerize with Docker\n\n## Structure\n- `app/`: API app code\n- `requirements.txt`: dependencies for easy setup\n\n## Results\nWorking REST endpoint that accepts input data and returns predictions.\n\n## Future Work\n- Add Swagger UI documentation\n- Deploy to cloud (Render, Heroku, or Azure)"
    },
    "Project_4_NLP_Feedback_Analysis": {
        "subfolders": ["notebooks"],
        "readme": "# 💬 NLP for User Feedback Analysis\n\n## Overview\nThis project uses natural language processing to analyze user reviews and feedback from gaming platforms.\n\n## Goals\n- Preprocess raw text (tokenization, lemmatization, stopword removal)\n- Perform sentiment analysis and topic modeling\n- Visualize key terms and clusters (word clouds, t-SNE)\n\n## Structure\n- `notebooks/`: NLP analysis notebooks\n\n## Results\nCharts and summaries of review sentiment, common complaints, praise areas.\n\n## Future Work\n- Integrate deep learning (e.g., BERT)\n- Automate review collection and dashboard reporting"
    }
}

# Create folders and write READMEs
for project, content in structure.items():
    project_path = os.path.join(ROOT_DIR, project)
    os.makedirs(project_path, exist_ok=True)

    for subfolder in content["subfolders"]:
        os.makedirs(os.path.join(project_path, subfolder), exist_ok=True)

    with open(os.path.join(project_path, "README.md"), "w", encoding="utf-8") as f:
        f.write(content["readme"])

# Create top-level README
top_readme = f"""# 📊 Data Science Portfolio

Welcome to my Data Science portfolio — a collection of projects showcasing my skills in data analysis, machine learning, and model deployment, with a focus on applications relevant to the iGaming and entertainment industries.

## 🧠 Projects
- [Project 1: Sports Outcome Prediction](./Project_1_Sports_Prediction)
- [Project 2: Customer Segmentation](./Project_2_Customer_Segmentation)
- [Project 3: ML Model Deployment](./Project_3_Model_Deployment)
- [Project 4: NLP Feedback Analysis](./Project_4_NLP_Feedback_Analysis)

## ⚙️ Tools & Skills
- Python (pandas, NumPy, scikit-learn)
- Data Cleaning, EDA, ML Modeling, APIs
- Jupyter, Flask/FastAPI, GitHub

📬 Contact: [LinkedIn](#) • your.email@example.com
"""

with open(os.path.join(ROOT_DIR, "README.md"), "w", encoding="utf-8") as f:
    f.write(top_readme)

print("✅ Portfolio folder structure created successfully!")
