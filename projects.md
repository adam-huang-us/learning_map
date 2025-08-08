

### **1. Underwriting Model Development**

* **Primary Goal**: Develop an underwriting scoring model to classify auto loans based on their survival rate.
* **Solution**: Pulled loan performance data from MySQL and implemented data cleaning with Python. Tested feature significance using Kaplan-Meier and Cox Proportional Hazards models. Developed and compared several survival models, including Accelerated Failure Time (AFT), Random Survival Forest (RSF), and a Neural MTLR model to **stratify customers into five distinct risk tiers**. After validating performance against historical Charge-off Net Loss (CNL), the RSF model was selected and deployed into production.
* **Result**: The new model boosted loan acceptance by 21%, leading to an additional $12 million in loan volume annually while maintaining risk thresholds.

***

### **2. Effective Call Prediction**

* **Primary Goal**: Prioritize the daily call list to increase contact efficiency and reduce operational costs.
* **Solution**: Extracted and cleaned customer contact history from structured and unstructured databases using Python. Engineered new features, such as 'optimal time to call' and 'days since last contact'. **Developed a stacked classification model to predict the likelihood of a successful contact.** The stacking ensemble consisted of a **Logistic Regression and Gradient Boosting Machine as base learners, with a final Random Forest meta-classifier** to generate the final propensity score.
* **Result**: Increased the **successful contact rate** from 18% to 45%, allowing the call center to connect with more customers with fewer attempts.

***

### **3. Delinquency Risk Prediction**

* **Primary Goal**: Predict the likelihood of 30+ day delinquency to enable proactive risk mitigation.
* **Solution**: Developed a stacked ensemble model in Python to predict delinquency. The first layer generated predictions from **three parallel base models: Logistic Regression, Random Forest, and XGBoost**. A second-layer **LightGBM model then aggregated these outputs** to produce a final, highly accurate risk score.
* **Result**: The model allowed for targeted interventions that helped reduce the overall portfolio delinquency rate from 18% to 15%.

***

### **4. OCR Typo and Web Scraped Text Correction**

* **Primary goal**: Filter out poor-quality questions from repositories caused by OCR or web-crawling bugs.
* **Solution**: Generated synthetic labeled data using NLP augmentation techniques like random cropping, random swap, random deletion, and shape-near word replacement. Fine-tuned an ALBERT (a BERT variant) model with a custom loss function that combined classification and token-level error prediction.
* **Result**: Achieved a 78% F1-score on the test set and helped filter out 5% of poor-quality questions from a repository of 3 million.

***

### **5. OCR System to Digitize Documents**

* **Primary Goal**: Build an automated OCR pipeline to reduce manual data entry costs and improve efficiency.
* **Solution**: Developed a multi-stage OCR pipeline. First, **OpenCV was used for document preprocessing and layout analysis to identify columns and tables**. Next, a U-Net model detected individual text blocks within these regions. A CRNN model then performed text recognition on each block. Finally, a Bi-LSTM model was used **for named-entity recognition (NER) to classify the extracted text** (e.g., 'Invoice Number', 'Client Name') before structuring the output as JSON.
* **Result**: Achieved 98% precision in text detection and 96% recall in text recognition.

***

### **6. Spam Detection in Educational Forum**

* **Primary Goal**: Increase spam blocking accuracy to over 98% and achieve annual cost savings of $30,000+.
* **Solution**: Used the Snorkel framework for weak supervision to programmatically generate a labeled dataset of 100,000 examples. **Developed and trained a text classification model using a Bi-LSTM architecture with an attention mechanism** to effectively identify and flag spam and NSFW content.
* **Result**: Achieved a 98.5% F1-score and successfully blocked over 50,000 spam posts per month.

***

### **7. Insurance Fraud Detection**

* **Primary Goal**: Deploy a model to a Web UI to predict if a customer is placing a fraudulent insurance claim.
* **Solution**: Consolidated data from multiple sources using MySQL. Performed extensive feature engineering in Python to handle missing values, scale numerical data, and encode categorical features. Trained and tuned XGBoost and SVM models using Grid Search to classify claims. Deployed the final model using a Flask API containerized with Docker.
* **Result**: The deployed XGBoost model **achieved an AUC score of 0.92 on the hold-out test set.** The Web UI successfully returns real-time fraud predictions, flagging potentially fraudulent claims for manual review.

***

### **8. Building Energy Consumption Prediction**

* **Primary Goal**: To predict a building’s energy consumption based on building types and climate parameters.
* **Solution**: Applied EDA and feature engineering with Python in Jupyter Notebook. Used LightGBM to predict the building energy consumption.
* **Result**: Achieved an RMSE of 17.72 (where the mean of the target feature, Energy Use Intensity, is 82.58).

***

### **9. NSFW Ad Pop-up Detection**

* **Primary goal**: Detect NSFW pop-up window screenshots and replace the current rule-based detector with a neural network model.
* **Solution**: Crawled NSFW images from search engines and manually labeled 10,000 images. Employed a fine-tuned VGG16 model to classify the images and deployed the service with Docker. Reviewed misclassifications and retrained the model on a monthly cycle.
* **Result**: Achieved a 99% F1-score. This new service became a core, high-value feature of the company’s product.

***

### **10. Applied Data Analysis in Price Negotiation**

* **Primary Goal**: To help the sales department conduct better price negotiations.
* **Solution**: Combined the client’s past 3+ years of quotation records with a raw material market index to discover the underlying logic of historical pricing. Created an interactive dashboard in Power BI to visualize these findings for the sales team.
* **Result**: Enabled the sales team to avoid unnecessary price concessions when raw material costs decreased, saving over $300,000 in profit annually.
