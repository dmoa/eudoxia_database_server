from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager
import secrets

from search import *


search = init_search_engine(last_embedded=True, embedding_name="dump")
app = Flask(__name__)


@app.route('/protected/search/', methods=['POST'])
def protected_search():
    response = {}
    response["message"] = "Search request received"
    response["search_results"] = search(request.form["search_input"])
    return jsonify(response)


@app.route('/protected/getitem', methods=['POST'])
def get_item():
    response = {}
    response["message"] = "Get item request received"

    item_name = request.form["item_name"]
    item_found = False

    for company_path in ["data/" + company_name for company_name in os.listdir("data")]:
        path = company_path + "/" + item_name + ".json"
        if os.path.exists(path):
            obj = get_product(path)
            response["item"] = obj
            item_found = True
            break

    if not item_found:
        response["error"] = "Item not found"
        return jsonify(response), 404

    return jsonify(response), 200


jwt = JWTManager(app)
app.run(ssl_context='adhoc', port=8080, debug=True, use_reloader=False)
