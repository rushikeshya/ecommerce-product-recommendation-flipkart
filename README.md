# 🛒 Flipkart Product Recommender System

A machine learning-based recommendation system that suggests relevant products using content-based and collaborative filtering techniques. Built with Python, Scikit-learn, and Flask, and designed to enhance user shopping experience.

## 🚀 Features

- 🔍 Content-Based Filtering: Recommends similar products based on features like price, category, rating, and popularity.
- 👥 Collaborative Filtering: Suggests products to users based on behavior patterns of similar users.
- 🧠 Preprocessing Pipeline: Cleans, encodes, and normalizes Flipkart product data.
- 💻 Web Application: Interactive Flask app with product detail pages and personalized recommendations.
- 📊 Visual Ready: HTML templates for intuitive product browsing and suggestions.

## 📂 Project Structure

flipkart-recommendation-system/<br>
│<br>
├── app.py                 # Flask application<br>
├── data_preparation.ipynb    # Data cleaning and preprocessing<br>
├── recommendation_models.ipynb # Recommendation algorithms<br>
├── processed_products.csv # Processed data<br>
├── content_based_model.pkl # Saved content-based model<br>
├── collab_filter_model.pkl # Saved collaborative filtering model<br>
├── requirements.txt       # Dependencies<br>
├── Procfile               # Heroku deployment config<br>
│<br>
├── templates/<br>
│   ├── index.html         # Home page<br>
│   ├── recommendations.html # Recommendations page<br>
│   └── product_detail.html # Product detail page<br>
│<br>
└── venv/                  # Virtual environment

## 🧪 How to Run
1. **Clone the repository:**
   ```bash
   git clone https://github.com/rushikeshya/flipkart-product-recommender.git
   cd flipkart-product-recommender
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Flask app:**
   ```
   python data_preparation.ipynb
   python recommendation_models.ipynb
   python app.py
   ```
4. Open in browser: Navigate to http://127.0.0.1:5000/ to interact with the app.

## 📦 Requirements 
   ```bash
   pip freeze > requirements.txt
   ```
Or use these common packages:

   ```txt
   pandas
   numpy
   scikit-learn
   scipy
   flask
   matplotlib
   seaborn
   ```
##🧑‍💻 Author
Made with ❤️ by **Rushikesh Yadav**
