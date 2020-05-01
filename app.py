from flask import Flask, render_template, url_for, request, redirect, flash, session
from werkzeug.utils import secure_filename
import pandas as pd
import uuid #Random Short Id
import os

UPLOAD_FOLDER = 'static/uploads/kmeans' #Location is saving uploaded
ALLOWED_EXTENSIONS = {'csv'} #Kind of file

app = Flask(__name__)

app.secret_key = b'_5#y2L"F4Q8z\n\xec]/' #Secret key of Upload
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/kmeans')
def kmeans_index():
    return render_template('kmeans/index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/kmeans/data', methods=['GET', 'POST'])
def kmeans_data():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            errors = 'No file part! Please choose 1 file csv !'
            return render_template('kmeans/data.html', errors=errors)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            errors = 'No selected file'
            return render_template('kmeans/data.html', errors=errors)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4())[:8] + '_' + filename)
            file.save(file_path)

            session['csvfile1'] = file_path #Save path file to session
            data = pd.read_csv(file_path)

            return render_template('kmeans/data.html', data=data.to_html(classes='table table-striped', header=False, index=False))

if __name__ == '__main__':
    app.run(debug=True)