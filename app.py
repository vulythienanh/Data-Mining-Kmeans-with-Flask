from flask import Flask, render_template, url_for, request, redirect, flash, session
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import uuid #Random Short Id
import os
import seaborn as sns
import matplotlib
matplotlib.use('template')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random

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

            session['csvfile'] = file_path #Save path file to session
            data = pd.read_csv(file_path)
        
            m = data.shape[1]

            return render_template('kmeans/data.html', data=data.to_html(table_id='myTable', classes='table table-striped', header=True, index=False), m=m)

@app.route('/kmeans/elbow', methods=['GET', 'POST'])
def kmeans_elbow():
    file_path = session.get('csvfile')
    data = pd.read_csv(file_path)

    col = request.form.getlist('cot') #Get values of checkbox form from
    col = np.array(col)
    col1 = col[0]
    col2 = col[1]

    session['col1'] = col1 #Save column to session
    session['col2'] = col2 #Save column to session

    m = data.shape[1]
    haha = 0
    X = data.iloc[int(haha):, [int(col1), int(col2)]].values

    # Tiến hành gom nhóm (Elbow)
    # Chạy thuật toán KMeans với k = (1, 10)

    clusters = []
    for i in range(1, 10):
        km = KMeans(n_clusters=i).fit(X)
        clusters.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(x=list(range(1, 10)), y=clusters, ax=ax)

    ax.set_title("Đồ thị Elbow")
    ax.set_xlabel("Số lượng nhóm")
    ax.set_ylabel("Gía trị Inertia")

    # plt.show()
    # plt.cla()
    image = 'static/images/kmeans/'+ str(uuid.uuid4())[:8] +'_elbow.png'
    plt.savefig(image)

    return render_template('kmeans/elbow.html', data=data.to_html(classes='table table-striped', header=False, index=False), url='/'+image)

@app.route('/kmeans/result', methods=['GET', 'POST'])
def kmeans_result():
    k = request.form.get('cluster')

    file_path = session.get('csvfile')
    data = pd.read_csv(file_path)

    col1  = session.get('col1')
    col2  = session.get('col2')
    # print(col1, col2)
    haha = 0

    X = data.iloc[int(haha):, [int(col1), int(col2)]].values

    print(X)

    km = KMeans(n_clusters=int(k))
    y_means = km.fit_predict(X)

    x_coordinates = [0, 1, 2, 3, 4, 5] #random color
    y_coordinates = [0, 1, 2, 3, 4, 5] #random color

    for i in range(0, int(k)):
        for x, y in zip(x_coordinates, y_coordinates): #random color
            rgb = (random.random(), random.random(), random.random())
        plt.scatter(X[y_means == i, 0], X[y_means == i, 1], s=100, c=[rgb], label='Nhóm ' + str(i+1))
    # plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s=100, c='red', label='Nhóm 1')
    # plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s=100, c='blue', label='Nhóm 2')
    # plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s=100, c='green', label='Nhóm 3')
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=100, c='yellow', label='Centeroid')
    # plt.plot(plt.legend()x,y)

    plt.style.use('fivethirtyeight')
    
    plt.title('K Means Clustering', fontsize=20)

    plt.xlabel('Age')
    plt.ylabel('Spending Score')

    plt.legend()
    plt.grid()
    image = 'static/images/kmeans/'+ str(uuid.uuid4())[:8] +'_plot.png'
    plt.savefig(image)

    return render_template('kmeans/result.html', data=data.to_html(classes='table table-striped', header=False, index=False), url='/'+image)

if __name__ == '__main__':
    app.run(debug=True)