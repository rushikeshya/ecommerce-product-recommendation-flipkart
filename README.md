# 🛒 Flipkart Product Recommender System

A machine learning-based recommendation system that suggests relevant products using content-based and collaborative filtering techniques. Built with Python, Scikit-learn, and Flask, and designed to enhance user shopping experience.

## 🚀 Features

- 🔍 Content-Based Filtering: Recommends similar products based on features like price, category, rating, and popularity.
- 👥 Collaborative Filtering: Suggests products to users based on behavior patterns of similar users.
- 🧠 Preprocessing Pipeline: Cleans, encodes, and normalizes Flipkart product data.
- 💻 Web Application: Interactive Flask app with product detail pages and personalized recommendations.
- 📊 Visual Ready: HTML templates for intuitive product browsing and suggestions.

## 📂 Project Structure

flipkart-recommendation-system/
│
├── app.py                 # Flask application
├── data_preparation.py    # Data cleaning and preprocessing
├── recommendation_models.py # Recommendation algorithms
├── processed_products.csv # Processed data
├── content_based_model.pkl # Saved content-based model
├── collab_filter_model.pkl # Saved collaborative filtering model
├── requirements.txt       # Dependencies
├── Procfile               # Heroku deployment config
│
├── templates/
│   ├── index.html         # Home page
│   ├── recommendations.html # Recommendations page
│   └── product_detail.html # Product detail page
│
└── venv/                  # Virtual environment

## 🧪 How to Run
1. **Clone the repository:**
   ```bash
   git clone https://github.com/rushikeshya/flipkart-product-recommender.git
   cd flipkart-product-recommender```
