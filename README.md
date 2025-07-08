📡 Network Slicing in 6G Networks
This project explores how machine learning models can classify services into the appropriate network slice types in future 6G networks, 
based on usage context and technical features. The models are trained to distinguish between eMBB, URLLC, and mMTC slices using real-world inspired datasets.

- Technologies Used
  - PyTorch (for MLP & CNN)
  - Scikit-learn (for Random Forest, XGBoost, LinearSVC)
  - XGBoost
  - SHAP (for explainability)

- Models & Evaluation
  - Deep Learning:
    - MLP (Multilayer Perceptron):
      -1 hidden layer with dropout and ReLU activation
    - CNN (1D):
      - Conv1D + pooling + flatten + dense

  - Machine Learning:
    - Random Forest
    - XGBoost
    - Linear SVC
  
  - Evaluation Techniques:
    - Train-Test Split
    - Stratified K-Fold Cross Validation
    - Classification Report: Precision, Recall, F1-score
  
- Explainable AI (XAI)
  - Using SHAP:
    - Visualizes feature importance per slice
    - Identifies which features most affect classification (e.g. LTE/5G, IoT, Delay)
