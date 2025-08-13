# Project Deliverable 3: Classification, Clustering, and Pattern Mining

## Sales Data Analysis: Advanced Machine Learning Techniques

###  Project Overview

This project implements comprehensive machine learning techniques on a sales dataset including classification models, clustering analysis, and association rule mining. The analysis provides actionable insights for business decision-making and customer behavior understanding using real sales transaction data with 1,000 records across 14 features.

###  Objectives

- **Classification**: Build and compare multiple classification models to predict high-value sales
- **Hyperparameter Tuning**: Optimize model performance using systematic parameter search
- **Clustering**: Identify customer segments using unsupervised learning techniques  
- **Pattern Mining**: Discover associations between products, customers, and sales patterns
- **Business Insights**: Extract actionable intelligence for real-world applications

###  Dataset Description

**Sales Dataset Features:**
- **Records**: 1,000 sales transactions
- **Features**: 14 columns including Product_ID, Sale_Date, Sales_Rep, Region, Sales_Amount, Quantity_Sold, Product_Category (Furniture, Food, Clothing, Electronics), Unit_Cost, Unit_Price, Customer_Type (Returning, New), Discount, Payment_Method (Cash, Bank Transfer, Credit Card), Sales_Channel (Online, Retail), and Region_and_Sales_Rep
- **Target Variables**: High_Sales (binary classification based on median sales amount) and Customer_Segment (categorical: Low, Medium, High)

###  Methodology

#### 1. Classification Models
- **Decision Tree Classifier**: Interpretable model with feature importance analysis
- **k-Nearest Neighbors (k-NN)**: Instance-based learning approach (k=5)
- **Naive Bayes**: Probabilistic classifier based on Bayes' theorem

#### 2. Hyperparameter Tuning
- **Grid Search Cross-Validation**: Systematic parameter optimization for Decision Tree
- **Parameters Tuned**: max_depth [3,5,7,10,None], min_samples_split [2,5,10], min_samples_leaf [1,2,4], criterion ['gini','entropy']
- **Evaluation Metric**: F1-score for balanced performance assessment
- **Best Parameters**: {'criterion': 'gini', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2}

#### 3. Model Evaluation
- **Confusion Matrix**: Classification accuracy breakdown for all models
- **ROC Curves**: True positive vs. false positive rate analysis with AUC scores
- **Performance Metrics**: Accuracy, F1-score, Precision, Recall for comprehensive evaluation
- **Feature Importance**: Identification of key predictive variables

#### 4. Clustering Analysis
- **K-Means Clustering**: Partition-based clustering with elbow method (optimal k=4)
- **Hierarchical Clustering**: Agglomerative clustering for comparison (4 clusters)
- **DBSCAN**: Density-based clustering for noise detection (eps=0.5, min_samples=5)
- **Evaluation**: Silhouette score analysis and PCA visualization

#### 5. Association Rule Mining
- **Apriori Algorithm**: Frequent itemset mining (min_support=0.1)
- **Association Rules**: Discovery of item relationships (min_confidence=0.5)
- **Metrics**: Support, Confidence, Lift analysis
- **Transaction Encoding**: Binary matrix creation from categorical and discretized numerical features

###  Key Findings

#### Classification Results
- **Best Model**: Tuned Decision Tree Classifier
- **Performance Metrics**:
  - **Base Decision Tree**: Accuracy: 54.00%, F1-Score: 53.06%
  - **k-NN**: Accuracy: 53.50%, F1-Score: 51.31%
  - **Naive Bayes**: Accuracy: 57.50%, F1-Score: 58.94%
  - **Tuned Decision Tree**: Accuracy: 55.00%, F1-Score: 62.81%
- **Key Feature Importance** (Tuned Decision Tree):
  - **Discount**: 54.2% (most important predictor)
  - **Unit_Price**: 23.6% (second most important)
  - **Quantity_Sold**: 15.4% (third most important)
- **Hyperparameter Impact**: Improved F1-score by 18.4% (from 53.06% to 62.81%)

#### Clustering Insights
- **Optimal Clusters**: 4 distinct customer segments identified using elbow method

- **Customer Segment Characteristics**:
  - **Cluster 0**: Avg Sales = $2,456.41, Avg Quantity = 27.6 (High-volume, low-value)
  - **Cluster 1**: Avg Sales = $4,961.50, Avg Quantity = 12.2 (Premium buyers)
  - **Cluster 2**: Avg Sales = $4,695.71, Avg Quantity = 37.9 (High-volume, high-value)
  - **Cluster 3**: Avg Sales = $7,727.25, Avg Quantity = 24.4 (Ultra-premium segment)

#### Association Rule Mining Results
- **Frequent Itemsets**: Successfully identified with minimum support threshold of 10%
- **Association Rules**: Discovered meaningful patterns with confidence ≥ 50%
- **Top Associations** (by Lift):
  - **Product_Category_Furniture → Customer_Type_Returning** (Lift: 1.13)
  - **Sales_Channel_Retail + Payment_Method_Credit Card → Customer_Type_Returning** (Lift: 1.11)  
  - **Product_Category_Electronics → Sales_Channel_Online** (Lift: 1.10)

###  Practical Applications & Business Relevance

#### 1. **Sales Prediction & Revenue Optimization**
- **Discount Strategy**: The model reveals discount is the strongest predictor (54.2% importance) of high sales, enabling data-driven discount optimization
- **Pricing Intelligence**: Unit price importance (23.6%) suggests strategic pricing adjustments can significantly impact sales outcomes
- **Quantity Forecasting**: Quantity sold importance (15.4%) supports inventory management and demand planning

#### 2. **Customer Segmentation & Targeted Marketing**
- **Premium Customer Identification**: Cluster 3 ($7,727 avg sales) represents high-value customers for VIP treatment
- **Volume-based Campaigns**: Cluster 2 (37.9 avg quantity) ideal for bulk purchase incentives  
- **Retention Programs**: Clusters 0 & 1 may need different retention strategies based on purchase patterns

#### 3. **Cross-selling & Product Recommendations**
- **Furniture-Customer Loyalty**: Strong association between furniture purchases and returning customers (Lift: 1.13)
- **Channel-Payment Optimization**: Retail + Credit Card combination drives customer retention (Lift: 1.11)
- **Electronics-Online Strategy**: Electronics products naturally align with online channels (Lift: 1.10)

#### 4. **Operational Efficiency**
- **Channel Optimization**: Association rules guide optimal sales channel strategies
- **Payment Method Strategy**: Credit card preference in retail environments informs payment processing
- **Regional Sales Planning**: Customer segmentation enables region-specific strategies



###  Challenges Encountered & Solutions

#### 1. **Model Performance Limitations**
- **Challenge**: Initial classification models showed moderate performance (54-58% accuracy)
- **Root Cause**: High variance in sales data and potential feature engineering limitations
- **Solution**: 
  - Applied comprehensive hyperparameter tuning (improved F1 by 18.4%)
  - Used ensemble approach comparison across multiple algorithms
  - Focused on F1-score for balanced evaluation given equal class distribution

#### 2. **Clustering Validation**
- **Challenge**: Determining optimal number of clusters with moderate silhouette scores (~0.31)
- **Root Cause**: High-dimensional feature space and potential overlapping customer behaviors
- **Solution**:
  - Combined elbow method with silhouette analysis
  - Used PCA for visualization and validation
  - Applied multiple clustering algorithms for comparison

#### 3. **Association Rule Mining Sparsity**
- **Challenge**: Initial difficulty finding strong association rules
- **Root Cause**: High cardinality in categorical variables and continuous numerical features
- **Solution**:
  - Discretized numerical variables into meaningful categories
  - Adjusted support and confidence thresholds iteratively
  - Created meaningful transaction encodings using domain knowledge

#### 4. **Feature Engineering Complexity**
- **Challenge**: Balancing interpretability with predictive power
- **Root Cause**: Mixed data types and varying scales across features
- **Solution**:
  - Applied systematic label encoding for categorical variables
  - Used StandardScaler for numerical feature normalization
  - Maintained interpretable feature names for business stakeholders

#### 5. **Data Quality & Preprocessing**
- **Challenge**: Ensuring consistent data types and handling potential outliers
- **Root Cause**: Real-world data inconsistencies and varying scales
- **Solution**:
  - Implemented comprehensive data preprocessing pipeline
  - Applied robust scaling techniques
  - Used stratified sampling for train-test split

---

*This project demonstrates the practical application of machine learning techniques to real business problems, providing both technical rigor and actionable business intelligence.*
