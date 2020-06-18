from flask import Flask, request, Response
import os
import pandas as pd
from test import File


app = Flask(__name__)


@app.route("/file", methods=['POST'])
# @cross_origin()
def predictRouteClient():

    df = pd.read_csv(request.files.get("file"))
    df.save()

    return Response("successful")

    # return Response("File upload is done successfully")



if __name__ == "__main__":
    app.debug = True
    app.run()