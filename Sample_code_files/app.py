from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load data and models
df = pd.read_csv('processed_products.csv')
with open('content_based_model.pkl', 'rb') as f:
    content_rec = pickle.load(f)
with open('collab_filter_model.pkl', 'rb') as f:
    collab_rec = pickle.load(f)

@app.route('/')
def home():
    sample_products = df.sample(6).to_dict('records')
    return render_template('index.html', products=sample_products)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.form
    
    if 'product_id' in data:
        product_id = data['product_id']
        recommendations = content_rec.recommend_products(product_id, n=5)
        return render_template('recommendations.html', 
                             recommendations=recommendations.to_dict('records'),
                             recommendation_type="Similar Products")
    
    elif 'user_id' in data:
        user_id = data['user_id']
        recommendations = collab_rec.recommend_products(user_id, n=5)
        return render_template('recommendations.html', 
                             recommendations=recommendations.to_dict('records'),
                             recommendation_type="Recommended For You")
    
    return "Invalid request", 400

@app.route('/product/<product_name>')
def product_detail(product_name):
    product = df[df['Product Name'] == product_name].iloc[0].to_dict()
    return render_template('product_detail.html', product=product)

if __name__ == '__main__':
    app.run(debug=True)