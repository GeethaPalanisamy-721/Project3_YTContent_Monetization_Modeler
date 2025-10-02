YouTube Content Monetization Modeler
A Machine Learning project that predicts YouTube Ad Revenue based on video performance metrics (views, likes, comments, subscribers, video length, etc.). The project demonstrates a complete ML lifecycle from EDA → preprocessing → feature engineering → model training → evaluation → deployment with Streamlit.

Dataset Description: The dataset includes information about video views, likes, comments, subscribers, video duration, upload date, country of origin, and the target variable, ad revenue. Additional engineered features are created to capture deeper patterns, such as engagement rate, subscribers per view, views per subscriber, log-transformed video length, categorical length grouping, temporal features like year, month, day of week, and whether the video was uploaded on a weekend, as well as grouping countries into frequent and rare categories.

Technologies used: The project uses modern Python technologies including Pandas and NumPy for data manipulation, Matplotlib and Seaborn for visualization, and Scikit-learn and XGBoost for building predictive models. Pipelines are implemented for preprocessing, including imputing missing values, scaling numerical features, and encoding categorical variables. Models compared include Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, and XGBoost, with evaluation based on metrics such as RMSE, MAE, and R². The best-performing model is saved as a serialized pipeline using Joblib, and artifacts such as metrics and plots are stored for reproducibility.

Why This Project is Useful: It teaches how to structure a professional project repository, how to clean and prepare data effectively, how to evaluate and compare multiple models systematically, and how to expose the final model in a user-friendly application.

How Users Can Get Started with the Project: 1. Clone the repository: git clone https://github.com/GeethaPalanisamy-721/Project3_YTContent_Monetization_Modeler. 2. Open the Power point file ‘Insights on YT Ad Monetization Modeler’ to get some needful insights.

Where Users Can Get Help • Check the Issues section on this GitHub repository for common problems. • For specific questions, open a new issue with detailed descriptions. • You can reach out via email at: geethabalan96@gmail.com or connect on LinkedIn: https://www.linkedin.com/in/geetha-palanisamy-777b52119

This project is maintained by: Geetha Palanisamy – Data Analyst focusing on Machine learning and dashboards. Contributions are welcome from data enthusiasts, developers, and industry experts. 

