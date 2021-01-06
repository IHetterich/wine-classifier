import pickle

from flask import Flask, render_template, request
from prediction import recommend

app = Flask(__name__)

# Unpickle model and vectorizer here so we only have to do it once.
model = pickle.load(open('pickles/model.pkl', 'rb'))
vectorizer = pickle.load(open('pickles/text_vec.pkl', 'rb'))

# Landing page with option to write in tasting notes.
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html', current_page='HOME')

# Recommendation page based on input from landing page.
@app.route('/recs', methods=['GET', 'POST'])
def recs():
    notes = request.form["writing_sample"]
    wines = recommend(model, vectorizer, notes)
    return render_template('recs.html', current_page='RECS', text=wines)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)