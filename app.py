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
    app.run(debug=True, port=5000)
