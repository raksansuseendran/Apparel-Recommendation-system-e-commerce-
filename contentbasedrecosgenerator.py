import redis
import json
from bs4 import BeautifulSoup
import urllib.request as urllib
import time
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np
import re

# Initialize a global Doc2Vec model variable
d2v_model = None

def train_doc2vec_model(redis_client):
    global d2v_model  # Use the global model variable

    # If the model hasn't been initialized yet, create a new one
    if d2v_model is None:
        d2v_model = Doc2Vec(vector_size=100, window=5, min_count=1, dm=1, epochs=20)
        # Initialize a list to store tagged documents
        tagged_documents = []

        # Load product data from Redis and create TaggedDocuments
        for key in redis_client.keys("product:*"):
            product_id = key.split(":")[-1]
            product_json = redis_client.get(key)
            product_data = json.loads(product_json)

            # TaggedDocument for description
            description_words = product_data["description"].split()
            tagged_description = TaggedDocument(description_words, [f"{product_id}_description"])

            # TaggedDocument for tags
            tags_words = product_data["tags"]
            tagged_tags = TaggedDocument(tags_words, [f"{product_id}_tags"])

            # TaggedDocument for color
            color_words = extract_color_from_description(product_data["description"])
            tagged_color = TaggedDocument(color_words.split(), [f"{product_id}_color"])

            # TaggggedDocument for title
            title_words = product_data["title"].split()
            tagged_title = TaggedDocument(title_words, [f"{product_id}_title"])


            tagged_documents.append(tagged_title)
            tagged_documents.append(tagged_tags)
            tagged_documents.append(tagged_description)
            tagged_documents.append(tagged_color)

        # Build the model vocabulary initially
        d2v_model.build_vocab(tagged_documents)

    else:
        # If the model has been initialized, update it iteratively
        tagged_documents = []

        # Load new product data from Redis and create TaggedDocuments
        for key in redis_client.keys("product:*"):
            product_id = key.split(":")[-1]
            product_json = redis_client.get(key)
            product_data = json.loads(product_json)

            # TaggedDocument for description
            description_words = product_data["description"].split()
            tagged_description = TaggedDocument(description_words, [f"{product_id}_description"])

            # TaggedDocument for tags
            tags_words = product_data["tags"]
            tagged_tags = TaggedDocument(tags_words, [f"{product_id}_tags"])

            # TaggedDocument for color
            color_words = extract_color_from_description(product_data["description"])
            tagged_color = TaggedDocument(color_words.split(), [f"{product_id}_color"])

            # TaggggedDocument for title
            title_words = product_data["title"].split()
            tagged_title = TaggedDocument(title_words, [f"{product_id}_title"])

            tagged_documents.append(tagged_title)
            tagged_documents.append(tagged_tags)
            tagged_documents.append(tagged_description)
            tagged_documents.append(tagged_color)

        # Update the model iteratively
        d2v_model.build_vocab(tagged_documents, update=True)
        d2v_model.train(tagged_documents, total_examples=len(tagged_documents), epochs=1)

    return d2v_model



def extract_color_from_description(description):
    # Example: "Color: Dark Navy"
    color_prefix = "Color:"
    color_index = description.find(color_prefix)
    if color_index != -1:
        color_start = color_index + len(color_prefix)
        color = description[color_start:].strip()
        # Remove any text after the next newline
        newline_index = color.find("\n")
        if newline_index != -1:
            color = color[:newline_index].strip()
        
        return color
    else:
        return ""

def clean_text(text):
    # Remove new lines, non-alphanumeric characters, and extra spaces
    cleaned_text = re.sub(r'\n', ' ', text)
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned_text)
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text

def find_related_products_advanced_nlp(description, product_type, product_id, tags, redis_client,title):
    global d2v_model  # Use the global model variable

    # Extract color from the description
    color = extract_color_from_description(description)

    # Clean the description text
    description = clean_text(description)



    # Create TaggedDocuments for description, color, and tags
    tagged_title = TaggedDocument(title.split(), [product_id])
    tagged_description = TaggedDocument(description.split(), [product_id])
    tagged_color = TaggedDocument(color.split(), [product_id])
    tagged_tags = TaggedDocument(tags, [product_id])

    # Infer document vectors for the input product
    title_vector = d2v_model.infer_vector(tagged_title.words)
    description_vector = d2v_model.infer_vector(tagged_description.words)
    color_vector = d2v_model.infer_vector(tagged_color.words)
    tags_vector = d2v_model.infer_vector(tagged_tags.words)

    # Calculate cosine similarity between the input product and all products
    similarities = []
    for key in redis_client.keys("product:*"):
        other_product_id = key.split(":")[-1]

        if other_product_id != str(product_id):
            product_json = redis_client.get(key)
            other_product = json.loads(product_json)

            if other_product["type"] == product_type:
                other_description_vector = d2v_model.infer_vector(clean_text(other_product["description"]).split())

                # Extract color from the other product's description
                other_color = extract_color_from_description(other_product["description"])
                tagged_other_color = TaggedDocument(other_color.split(), [other_product_id])
                other_color_vector = d2v_model.infer_vector(tagged_other_color.words)
                

                other_tags_vector = d2v_model.infer_vector(other_product["tags"])

                other_title_vector = d2v_model.infer_vector(other_product["title"].split())

                # Calculate cosine similarity between description vectors
                description_similarity = np.dot(description_vector, other_description_vector) / (
                        np.linalg.norm(description_vector) * np.linalg.norm(other_description_vector))

                # Calculate cosine similarity between color vectors
                color_similarity = np.dot(color_vector, other_color_vector) / (
                        np.linalg.norm(color_vector) * np.linalg.norm(other_color_vector))

                # Calculate cosine similarity between tags vectors
                tags_similarity = np.dot(tags_vector, other_tags_vector) / (
                        np.linalg.norm(tags_vector) * np.linalg.norm(other_tags_vector))
                
                # Calculate cosine similarity between title vectors

                title_similarity = np.dot(title_vector, other_title_vector) / (
                        np.linalg.norm(title_vector) * np.linalg.norm(other_title_vector))

                # Combine similarity scores, giving equal weight to description, color, and tags
                combined_similarity = (
                        0.3* description_similarity + 0.2* color_similarity + 0.3* tags_similarity + 0.2* title_similarity
                )

                similarities.append((other_product_id, combined_similarity))

    # Sort products by similarity score in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    ml_recommendations = []
    # Return top 10 most similar products
    for product_id, _ in similarities[:25]:
        key = f"product:{product_id}"
        product_json = redis_client.get(key)
        product_data = json.loads(product_json)
        ml_recommendations.append(product_data)

    # print the handles of the ml_recommendations
    # ml_recommendations_handle = [product['handle'] for product in ml_recommendations]
    # print(ml_recommendations_handle)

    return ml_recommendations


# Initialize Redis clients, load models, and process products
redis_host = "localhost"
redis_port = 6379
redis_password = None

redis_client = redis.StrictRedis(
    host=redis_host,
    port=redis_port,
    db=3,
    password=redis_password,
    decode_responses=True
)

redis_client2 = redis.StrictRedis(
    host=redis_host,
    port=redis_port,
    db=4,
    password=redis_password,
    decode_responses=True
)

with open('config.json') as config_file:
    config = json.load(config_file)

if config['attemptkey'] == "new":
    print("Redis DB reset")
    y_json_file = config['json_file']

    with open(y_json_file, 'r', encoding='utf-8') as json_file:
        json_data = json_file.read()

    json_data = json.loads(json_data)

    for product in json_data:
        if 'description' in product and product['description'] is not None:
            # Use BeautifulSoup to remove HTML tags from the description
            product['description'] = BeautifulSoup(product['description'], features="html.parser").get_text()

        product['handle'] = f"{config['base_url']}products/{product['handle']}"
        product_id = product['id']
        redis_key = f'product:{product_id}' 

        redis_client.set(redis_key, json.dumps(product, indent=4))
    
    d2v_model = train_doc2vec_model(redis_client)

    # Save the Doc2Vec model in the working directory
    d2v_model.save("doc2vec.model")
else:
    pass

# Load the Doc2Vec model from the working directory
d2v_model = Doc2Vec.load("doc2vec.model")

product_list = []

# Retrieve all keys containing product details
all_keys = redis_client.keys('product:*')

redis_client.close()

# Create a list to store product details for rendering
products = []

# Iterate through the keys and fetch product details
for redis_key in all_keys:
    product_details_json = redis_client.get(redis_key)
    if product_details_json:
        product_details = json.loads(product_details_json)
        products.append(product_details)


product_images = {}

for product_details in products:
    product_id = product_details['id']
    
   # product_images dict should have product_id as main key and list of images and product type as value
   #type should be subordinate key for acessing the product type, similarly for featured image

    product_images[product_id] = {}

    product_images[product_id]['type'] = product_details['type']

    #featured image

    product_images[product_id]['featured_image'] = product_details['featured_image']


  
# -------  have all images in product_images dict
# 

# Fetch Shopify recommendations
for product_details in products:
    base_url = config['base_url']
    # Fetch Shopify recommendations
    url = f"{base_url}recommendations/products.json?product_id={product_details['id']}&intent=related"
    print(url)
    data = urllib.urlopen(url).read()
    shopify_products = json.loads(data)['products']


    for rec_product in shopify_products:
        # Add a new key to product_details
        if 'shopify_recommendations' not in product_details:
            product_details['shopify_recommendations'] = []

        product_details['shopify_recommendations'].append({
            'id': rec_product['id'],
            'title': rec_product['title'],
            'handle': f"{base_url}products/{rec_product['handle']}",
            # Use BeautifulSoup to remove HTML tags from the description
            'description': BeautifulSoup(rec_product['description'], features="html.parser").get_text(),
            'type': rec_product['type'],
            'tags': rec_product['tags'],
            'price': rec_product['price'] / 100,
            'featured_image': 'https:' + rec_product['featured_image'] if rec_product['featured_image'] else None
        })
    time.sleep(15)

    # knn_recommendations
    ml_recommendations = []

    try:
        output = find_related_products_advanced_nlp(product_details["description"], product_details["type"], product_details["id"], product_details["tags"], redis_client,product_details["title"])

        product_details['ml_recommendations'] = output

    except Exception as e:
        print(e)
        print("Error executing command")

    product_list.append(product_details)

    # Save the product details list to a JSON file
    with open('product_details.json', 'w') as outfile:
        json.dump(product_list, outfile, indent=4)

    # Store the product details in Redis with a unique key (e.g., based on the product ID)
    for product_details in product_list:
        product_id = product_details['id']
        redis_key = f'product:{product_id}'

        # Store the JSON data in Redis with an expiration time if needed
        redis_client2.set(redis_key, json.dumps(product_details, indent=4))
    # You can set an expiration time (in seconds) like this:
    # redis_client.setex(redis_key, json.dumps(product_details, indent=4), expiration_time_in_seconds)

# Close the Redis connection when you're done
redis_client2.close()