"""
OpeningsMatch - Flask demo UI.

Run:  python app.py
Then: http://localhost:5000
"""

from flask import Flask, render_template, request
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from recommender import OpeningsMatch, Door  # noqa: E402

app = Flask(__name__)
engine = OpeningsMatch()


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    door = Door()

    if request.method == "POST":
        door = Door(
            fire_rated=request.form.get("fire_rated") == "on",
            is_egress=request.form.get("is_egress") == "on",
            handed=request.form.get("handed") == "on",
            acoustic=request.form.get("acoustic") == "on",
            security_level=int(request.form.get("security_level", 2)),
            location=request.form.get("location", "interior"),
            traffic=request.form.get("traffic", "standard"),
        )
        result = engine.recommend_set(door)

    return render_template("index.html", result=result, door=door,
                           catalog_size=len(engine.catalog))


if __name__ == "__main__":
    # Railway (and most PaaS) inject the port to listen on via $PORT.
    # Bind to 0.0.0.0 so the platform's router can reach the container -
    # 127.0.0.1 only accepts connections from INSIDE the container, which is
    # why the deploy succeeded but the app never responded.
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
