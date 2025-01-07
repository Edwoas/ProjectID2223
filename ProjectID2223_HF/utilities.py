from models import *
from transformers import AutoTokenizer, AutoModel
import http.client
import gradio as gr
import torch
import os
import re
import json

global conn, headers
log_messages = []

# Load tokenizer and distil bert model
roberta_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
roberta_model = AutoModel.from_pretrained("distilbert-base-uncased")
model_dir = get_model_from_hopsworks("ID2223Lab1", "fine_tune_bert_un", 1)
model_path = os.path.join(model_dir, "fine_tune_bert_un.pth")
checkpoint = torch.load(model_path, map_location="cpu")
classifier = ClassifierHead(768, 5)
classifier.load_state_dict(checkpoint)


def add_log_message(message):
    log_messages.append(message)
    if len(log_messages) > 100:
        log_messages.pop(0)
    return "\n".join(log_messages)


def predict(texts, model, classifier_head, tokenizer):
    # Tokenize input texts
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Compute CLS embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # Extract CLS token

    # Pass embeddings to classifier head
    with torch.no_grad():
        logits = classifier_head(cls_embeddings)  # Get logits
        predictions = torch.argmax(logits, dim=1)  # Predicted class
    return predictions


def api_connection():

    global conn, headers

    conn = http.client.HTTPSConnection("real-time-amazon-data.p.rapidapi.com")
    api_key = os.environ["RAPID_API_KEY"]

    headers = {
        f'x-rapidapi-key': f"{api_key}",
        'x-rapidapi-host': "real-time-amazon-data.p.rapidapi.com"
    }


def api_request(path):
    global conn, headers

    try:
        print("Processing: conn.request(GET, path, headers=headers) ...")
        conn.request("GET", path, headers=headers)
        print("Finished!")
        print("Processing: res = conn.getresponse() ...")
        res = conn.getresponse()
        print("Finished")

        if res.status == 200:  # Success
            data = res.read()
            json_data = data.decode("utf-8")
            json_data = json.loads(json_data)
            add_log_message(f"API Request Successful: {path}")
            return json_data
        elif res.status == 429:  # Rate limit exceeded
            print(f"Rate limited. Status: {res.status}")
            add_log_message(f"Rate limited. Status: {res.status}")
        elif res.status in [400, 403, 404]:  # Client-side errors
            print(f"Client error: {res.status} - {res.reason}")
            add_log_message(f"Client error: {res.status} - {res.reason}")
            return {"error": f"Client error: {res.reason}"}
        elif res.status >= 500:  # Server-side errors
            print(f"Server error: {res.status} - {res.reason}")
            add_log_message(f"Server error: {res.status} - {res.reason}")
            return {"error": f"Server error: {res.reason}"}
    except Exception as e:
        print(f"API request failed: {e}")
        add_log_message(f"API request failed: {e}")
        conn.close()  # Close the connection
        conn = http.client.HTTPSConnection("real-time-amazon-data.p.rapidapi.com", timeout=12)  # Reinitialize

        return {"error": str(e)}


def prompt_reviews(asin):

    reviews = []
    for page in range(1, 6):
        path = f"/product-reviews?asin={asin}&country=US&page={page}"
        response = api_request(path)

        if "error" in response:  # Handle API errors gracefully
            print(response["error"])
            return 0, 0

        page_reviews = response.get('data', {}).get("reviews", [])
        if len(page_reviews) == 0:
            continue

        for review in page_reviews:
            reviews.append(review["review_comment"])

    if not reviews:
        return 0, 0

    ratings = predict(reviews, model=roberta_model, tokenizer=roberta_tokenizer, classifier_head=classifier).tolist()
    avg_rating = sum(ratings) / len(ratings)
    squared_diffs = [(x - avg_rating) ** 2 for x in ratings]
    std_dev = (sum(squared_diffs) / (len(ratings) - 1)) ** 0.5
    return round(avg_rating, 2), round(std_dev, 2)


def prompt_product(query):
    query = re.sub(r'\s+', '%20', query)
    path = f"/search?query={query}&page=1&country=US&sort_by=RELEVANCE&product_condition=ALL"
    response = api_request(path)

    if "error" in response:  # Handle API errors gracefully
        print(response["error"])
        return {}

    filtered_products = {}
    for product in response.get('data', {}).get('products', []):
        if len(product["product_title"]) <= 71:
            filtered_products[product["product_title"]] = product["asin"]
            if len(filtered_products) > 15:
                break
    return filtered_products


def search_products(query):
    global product_mapping
    product_mapping = prompt_product(query)
    if not product_mapping:
        return gr.update(choices=["No results found or an error occurred."], value="No results found.")
    return gr.update(choices=list(product_mapping.keys()), value=None)


def confirm_selection(selected_product):
    if selected_product in product_mapping:
        asin = product_mapping[selected_product]
        avg_rating, std_dev = prompt_reviews(asin)
        if avg_rating == 0 and std_dev == 0:
            return "Error: Unable to fetch reviews or no reviews found.", None  # No chart in case of error
        return f"Average rating: {avg_rating}\nStandard deviation: {std_dev}", (avg_rating, std_dev)
    return "No product selected. Please choose a product.", None


# Function to generate a simple HTML visualization
def create_html_visualization(data):
    if data is None:
        return "No data available to visualize."

    avg_rating, std_dev = data
    scale_max = 5  # Scale from 1 to 5
    whisker_left = max(avg_rating - std_dev, 1)  # Ensure minimum scale of 1
    whisker_right = min(avg_rating + std_dev, scale_max)  # Ensure max scale of 5

    # Create a simple horizontal bar visualization with whiskers
    html = f"""
    <div style="display: flex; flex-direction: column; align-items: center;">
        <div style="width: 80%; height: 30px; background: lightgray; position: relative; border-radius: 5px;">
            <div style="width: {(avg_rating/scale_max)*100}%; height: 100%; background: skyblue; border-radius: 5px;"></div>
            <div style="position: absolute; top: -5px; left: {(whisker_left/scale_max)*100}%; height: 40px; width: 2px; background: black;"></div>
            <div style="position: absolute; top: -5px; left: {(whisker_right/scale_max)*100}%; height: 40px; width: 2px; background: black;"></div>
        </div>
        <p>Average Rating: {avg_rating:.2f} (±{std_dev:.2f})</p>
    </div>
    """
    return html


def process_selection(selected):
    result = confirm_selection(selected)  # Call once
    text_output = result[0]
    chart_output = create_html_visualization(result[1])  # Generate chart from the second output
    return text_output, chart_output


def process_search(query):
    # Call functions once and store their outputs
    search_result = search_products(query)
    log_message = add_log_message(f"Search triggered for: {query}")
    return search_result, log_message



