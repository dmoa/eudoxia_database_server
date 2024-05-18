from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager
import secrets

from search import *


search = init_search_engine()
app = Flask(__name__)


@app.route('/protected/search/', methods=['POST'])
def protected_search():
    response = {}
    response["message"] = "Search request received"
    response["search_result"] = search(request.form["search_input"])
    return jsonify(response)


jwt = JWTManager(app)
app.run(ssl_context='adhoc', port=80, debug=True, use_reloader=True)