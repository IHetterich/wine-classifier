import pickle

from flask import Flask, render_template, request

from recommender import Recommender

app = Flask(__name__)

# Unpickle our recommenderb so we only have to do it once.
model = pickle.load(open('pickles/recommender.pkl', 'rb'))

# Landing page with option to write in tasting notes.
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html', current_page='HOME')

# Recommendation page based on input from landing page.
@app.route('/recs', methods=['GET', 'POST'])
def recs():
    notes = request.form["writing_sample"]
    wines = model.predict(notes)
    return render_template('recs.html', current_page='RECS', text=wines)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)