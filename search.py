import os
import torch
import clip
from PIL import Image
import numpy as np
import base64
import json
import io
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from http.server import BaseHTTPRequestHandler, HTTPServer
import random

from util import *

set_signal_to_nothing()


# @TODO turn most of this into post processing
def get_product(object_json_path):
    obj = json.loads(read_entire_file(object_json_path))
    image_path = object_json_path[:-5] + "." + obj["extension"]
    obj["image_encoded"] = base64.b64encode(read_entire_file_rb(image_path)).decode('utf-8')
    url = base64.b64decode(os.path.splitext(os.path.basename(object_json_path))[0]).decode("utf-8")
    obj["url"] = url # @TODO this should be done in post process json

    # obj["image_url"] = f"data:image/jpeg;base64,{obj}"

    return obj


# gpt generated
def print_progress_bar(task_name, num_tasks_completed, total_tasks):
    length = 50
    fill = '█'
    prefix = task_name
    suffix = 'Complete'

    percent = ("{0:.1f}").format(100 * (num_tasks_completed / float(total_tasks)))
    filled_length = int(length * num_tasks_completed // total_tasks)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix}: |{bar}| {percent}% {suffix}', end='\r')
    # Print New Line on Complete
    if num_tasks_completed == total_tasks:
        print()

def encode_images(device, model, preprocess, objects):
    embeddings = []
    num_items = len(objects)
    processed = 0

    batch_images = []
    batch_size = 32

    with torch.no_grad():
        for obj in objects:

            image_data = base64.b64decode(obj["image_encoded"])
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            if image.mode == 'P':
                image = image.convert('RGBA')
            else:
                image = image.convert('RGB')

            image_tensor = preprocess(image).unsqueeze(0).to(device)

            batch_images.append(image_tensor)

            # Check if we have a full batch or are at the end of the list
            if len(batch_images) == batch_size or (processed + 1) == num_items:
                batch_tensor = torch.cat(batch_images, dim=0)
                embeddings_batch = model.encode_image(batch_tensor)
                embeddings.append(embeddings_batch)
                batch_images = []  # Reset the batch list

            processed += 1
            print_progress_bar("Embedding images", processed, num_items)

        print("Combining embeddings into a single tensor")
        embeddings = torch.cat(embeddings, dim=0)  # Combine into a single tensor
        print("Done")

    return embeddings

def init_search_engine(last_embedded=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using", device)
    model, preprocess = clip.load("ViT-L/14@336px", device=device)

    data_path = "data"
    company_names = os.listdir(data_path)
    company_paths = [data_path + "/" + company_name for company_name in company_names]

    # fact. image_paths must be an ordered array.
    # fact. indices will give us the image path from this.

    object_json_paths = []
    objects = []

    if last_embedded:
        if not os.path.exists("last_embedded.pt"):
            print("Error! Couldn't load previously embedded things")
            return

        print("Loading embeddings from last_embedded.pt")
        image_embeddings = torch.load("last_embedded.pt")

        object_json_paths = read_entire_file("last_embedded_paths.txt").split("\n")
        for object_json_path in object_json_paths:
            obj = get_product(object_json_path)
            objects.append(obj)
    else:
        for company_path in company_paths:
            paths = [company_path + "/" + path for path in os.listdir(company_path)]
            object_json_paths += [path for path in paths if path.endswith("json")]

        random.shuffle(object_json_paths)
        object_json_paths = object_json_paths[:250]

        write_entire_file_w("last_embedded_paths.txt", "\n".join(object_json_paths))

        for object_json_path in object_json_paths:
            obj = get_product(object_json_path)
            objects.append(obj)
        print("Creating new embeddings")
        image_embeddings = encode_images(device, model, preprocess, objects)
        torch.save(image_embeddings, "last_embedded.pt")


    def search(text):
        top_k = 50

        with torch.no_grad():
            text_encoded = clip.tokenize([text]).to(device)
            text_embedding = model.encode_text(text_encoded)

        similarities = (text_embedding @ image_embeddings.T).squeeze(0)

        # Calculate confidence scores (convert cosine similarities to percentage)
        scores = similarities.cpu().numpy()
        min_score = np.min(scores)
        max_score = np.max(scores)

        normalized_scores = 100 * (scores - min_score) / (max_score - min_score)

        top_k_indices = (-similarities).argsort().tolist()[:top_k]

        top_k_objects = []
        for index in top_k_indices:
            obj = objects[index]
            obj_with_confidence = obj.copy()  # Shallow copy to avoid modifying original objects
            obj_with_confidence["search_confidence"] = round(float(normalized_scores[index]), 1)
            top_k_objects.append(obj_with_confidence)

        return top_k_objects

    print("Search engine ready")
    return search