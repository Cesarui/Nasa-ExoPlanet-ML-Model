from Main import predict_exoplanet
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("index.html")
    else:
        depth = request.form.get("transit-depth")
        duration = request.form.get("transit-duration")
        period = request.form.get("orbital-period")
        pRad = request.form.get("planet-radius")
        sRad = request.form.get("stellar-radius")
        isExoplanet = predict_exoplanet(depth, duration, period, pRad, sRad)
        print(result)
        if isExoplanet:
            return redirect(url_for("result"))
        else:
            return redirect(url_for("result2"))
        
@app.route("/result", methods=["GET","POST"])
def result():
    return render_template("result.html")

@app.route("/result2", methods=["GET","POST"])
def result2():
    return render_template("result2.html")




if __name__ == "__main__":
    app.run()