from flask import Flask, render_template, request

import model as afr

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])

def home():
    pred1 = pred2 = 'No review yet'
    if request.method=='POST':
        user_input = request.form.get('review_text')
        print(user_input)
        user_input = afr.clean_text(user_input)
        pred1, pred2  = afr.predictText(user_input)
        print(pred1, pred2)
    return render_template('index.html', pred1=pred1, pred2 = pred2)

if __name__ == '__main__':
    app.run(debug=True)

