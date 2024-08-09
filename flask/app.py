#!/usr/bin/env python
# coding: utf-8

from crypt import methods
from flask import Flask, render_template, request, redirect, g, url_for, session, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager, login_user, logout_user, login_required, current_user
from flask_paginate import Pagination, get_page_parameter
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt   
from matplotlib.backends.backend_agg import FigureCanvasAgg
from datetime import datetime
import pytz
import os
import io
import urllib.parse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SECRET_KEY'] = os.urandom(24)
db = SQLAlchemy(app)

DATABASE = 'blog.db'
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect('blog.db')
    return g.db

login_manager = LoginManager()
login_manager.init_app(app)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(10), unique=True, nullable=False)
    password = db.Column(db.String(12), nullable=False)
    gender = db.Column(db.String(3), nullable=False)
    age = db.Column(db.Integer(), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now(), nullable=False)

class Favorite(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(10), nullable=False)
    manga_name = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.now(), nullable=False)

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                 endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@login_manager.unauthorized_handler
def unauthorized():
    return redirect('/login')

@app.route("/sitemap")
@app.route("/sitemap/")
@app.route("/sitemap.xml")
def sitemap():
    sitemaplist = []
    db = get_db()
    cur = db.execute(f'SELECT name FROM manga_data GROUP BY 1')
    sitemapmangas = cur.fetchall()
    for manga in sitemapmangas:
        sitemaplist.append(manga)
    xml_sitemap = render_template('sitemap.xml', sitemaplist=sitemaplist)
    response = make_response(xml_sitemap)
    response.headers["Content-Type"] = "application/xml"
    return response


@app.route("/")
def index():
    db = get_db()
    cur = db.execute("WITH t0 AS (SELECT product_title AS name, ROUND(AVG(review_score),1) AS score FROM review GROUP BY 1) SELECT DISTINCT * FROM t0  JOIN manga_data USING(name) ORDER BY score DESC LIMIT 15")
    mangas = cur.fetchall()
    cur = db.execute(f'SELECT * FROM review ORDER BY created_at DESC LIMIT 10')
    reviews = cur.fetchall()
    return render_template('index.html', mangas=mangas, reviews=reviews)

@app.route("/product/<manga>", methods=['GET', 'POST'])
def product_page(manga):
    db = get_db()
    cur = db.execute(f'SELECT * FROM manga_data WHERE name = "{manga}"')
    mangas = cur.fetchone()
    cur = db.execute(f'SELECT * FROM review WHERE product_title = "{manga}" ORDER BY created_at DESC')
    revies = cur.fetchall()
    cur = db.execute(f'SELECT ROUND(AVG(review_score),1) FROM review WHERE product_title = "{manga}"')
    revies_avgscore = cur.fetchone()
    manga_one = manga
    if current_user.is_authenticated:
        username = current_user.username
    else:
        username = '匿名さん' 
    cur = db.execute(f'SELECT DISTINCT manga_name FROM favorite WHERE username = "{username}"')
    favorite = cur.fetchall()
    favorite_list = []
    for a in favorite:
        for b in a:
            favorite_list.append(b)
    if request.method == 'POST':
        if current_user.is_authenticated:
            username = current_user.username
        else:
            username = '匿名さん' 
        review_score = request.form.get('title')
        body = request.form.get('body')
        cursor = db.cursor() 
        cursor.execute("INSERT INTO review(username, review_score, body, product_title) VALUES(?, ?, ?, ?)",(username, review_score, body, manga))
        db.commit()

        import tweepy
        # API情報を記入
        BEARER_TOKEN = os.getenv('BEARER_TOKEN')
        API_KEY = os.getenv('API_KEY')
        API_SECRET = os.getenv('API_SECRET')
        ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')
        ACCESS_TOKEN_SECRET = os.getenv('ACCESS_TOKEN_SECRET')


        # クライアント関数を作成
        def ClientInfo():
            client = tweepy.Client(bearer_token    = BEARER_TOKEN,
                                consumer_key    = API_KEY,
                                consumer_secret = API_SECRET,
                                access_token    = ACCESS_TOKEN,
                                access_token_secret = ACCESS_TOKEN_SECRET,
                                )
            
            return client
        # ★メッセージを指定
        if len(body) <= 90:
            message = f'''\
{manga}のレビュー 
スコア : {review_score}
レビュー内容 : {body}
https://manga-review.ml/
    '''
        else:
            body = body[0:90]+'...'
            message = f'''\
{manga}のレビュー 
スコア : {review_score}
レビュー内容 : {body}
https://manga-review.ml/
    '''


        # 関数
        def CreateTweet(message):
            tweet = ClientInfo().create_tweet(text=message)
            return tweet

        # 関数実行・結果出力
        CreateTweet(message)
        return redirect(f'/product/{manga}')
    else:
        return render_template('product.html', mangas=mangas, revies=revies, revies_avgscore=revies_avgscore, favorite_list=favorite_list, manga_one=manga_one)


@app.route('/result/')
def result():
    if request.args.get('search_word', '') is None:
        manga_list = []
        db = get_db()
        cur = db.execute(f'SELECT * FROM manga_data')
        manga_list = cur.fetchall()
        page = request.args.get(get_page_parameter(), type=int, default=1)

        # (2)１ページに表示させたいデータ件数を指定して分割(１ページに3件表示)
        PageData = manga_list[(page - 1)*20: page*20]

        # (3) 表示するデータリストの最大件数から最大ページ数を算出
        MaxPage = (- len(manga_list) // 20) * -1

        pagination = Pagination(page=page, total=len(manga_list), per_page=20, css_framework='bootstrap5')

        # (4) ページネーションに必要なデータをHTMLファイルに引き渡します。
        return render_template('result.html', pagination=pagination, PageData = PageData)
    else:
        search_word = request.args.get('search_word', '')
        manga_list = []
        db = get_db()
        cur = db.execute(f'SELECT * FROM manga_data WHERE name like "%{search_word}%" OR author like "%{search_word}%" OR genre like "%{search_word}%" OR publisher like "%{search_word}%" OR story like "%{search_word}%" OR publication_magazines like "%{search_word}%"')
        manga_list = cur.fetchall()
        page = request.args.get(get_page_parameter(), type=int, default=1)

        # (2)１ページに表示させたいデータ件数を指定して分割(１ページに3件表示)
        PageData = manga_list[(page - 1)*20: page*20]

        # (3) 表示するデータリストの最大件数から最大ページ数を算出
        MaxPage = (- len(manga_list) // 20) * -1

        pagination = Pagination(page=page, total=len(manga_list), per_page=20, css_framework='bootstrap5')

        # (4) ページネーションに必要なデータをHTMLファイルに引き渡します。
        return render_template('result.html', pagination=pagination, PageData = PageData, search_word=search_word)


@app.route("/signup", methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        gender = request.form.get('gender')
        mangas = request.form.getlist('favorite_manga')
        age = request.form.get('age')
        user = User(username=username, password=generate_password_hash(password, method='sha256'), gender=gender, age=age)
        db.session.add(user)
        db.session.commit()

        for manga in mangas:
            favorite = Favorite(username=username, manga_name=manga)
            db.session.add(favorite)
            db.session.commit()
        user = User.query.filter_by(username=username).first()
        if check_password_hash(user.password, password):
            login_user(user)
            return redirect('/mypage')
    else:
        mangas = []
        db_sqlite = get_db()
        cur = db_sqlite.execute("SELECT * FROM manga_data WHERE name IN ('ONE PIECE カラー版', '呪術廻戦', 'チェンソーマン', 'キングダム', 'GANTZ', '嘘喰い', '五等分の花嫁', '進撃の巨人', 'SPY×FAMILY', '君に届け リマスター版', '花より男子', 'NANA―ナナ―') ORDER BY created_at LIMIT 12")
        mangas = cur.fetchall()
        return render_template('signup.html', mangas = mangas)


@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        if user:
            if check_password_hash(user.password, password):
                login_user(user)
                return redirect('/mypage')
            else:
                error = 'passwordが違います'
                return render_template('login.html', error=error)
        else:
            error = 'usernameが違います'
            return render_template('login.html', error=error)
    else:
        if current_user.is_authenticated:
            return redirect('/mypage')
        else:
            return render_template('login.html')

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect('/login')

@app.route("/mypage")
@login_required
def mypage():
    username = current_user.username
    db = get_db()
    cur = db.execute(f'SELECT * FROM review WHERE username = "{username}" ORDER BY created_at DESC')
    review_list = cur.fetchall()
    cur = db.execute(f'SELECT manga_name,img FROM favorite LEFT JOIN manga_data ON favorite.manga_name = manga_data.name WHERE username = "{username}" GROUP BY 1,2')
    favorite_list = cur.fetchall()
    return render_template('mypage.html', review_list=review_list, favorite_list=favorite_list)


@app.route("/<review_id>/update", methods=['GET', 'POST'])
@login_required
def uodate(review_id):
    db = get_db()
    cur = db.execute(f'SELECT * FROM review WHERE review_id = {review_id}')
    review = cur.fetchall()
    if request.method == 'GET':
        cur = db.execute(f'SELECT username FROM review WHERE review_id = {review_id}')
        review_user = ''
        for item in cur.fetchone():
            review_user = review_user + item
        username = current_user.username
        if username == review_user:
            return render_template('update.html', review=review)
        else:
            return redirect('/login')
    else:
        new_score = request.form.get('title')
        new_body = request.form.get('body')
        cur = db.execute(f'update review set review_score = {new_score} where review_id = {review_id}')
        cur = db.execute(f'update review set body = "{new_body}" where review_id = {review_id}')
        cur2 = db.execute(f'SELECT product_title FROM review WHERE review_id = {review_id}')
        str = ''
        for item in cur2.fetchone():
            str = str + item
        db.commit()
        return redirect(f'/product/{str}')

@app.route("/<review_id>/delete", methods=['GET'])
@login_required
def delete(review_id):
    db = get_db()
    cur = db.execute(f'SELECT username FROM review WHERE review_id = {review_id}')
    review_user = ''
    for item in cur.fetchone():
        review_user = review_user + item
    username = current_user.username
    if username == review_user:
        cur2 = db.execute(f'SELECT product_title FROM review WHERE review_id = {review_id}')
        cur = db.execute(f'DELETE FROM review WHERE review_id = {review_id}')
        str = ''
        for item in cur2.fetchone():
            str = str + item
        db.commit()
        return redirect(f'/product/{str}')
    else:
        return redirect('/login')

@app.route("/<manga>/favorite")
@login_required
def favorite(manga):
    username = current_user.username
    created_at =datetime.now()
    db = get_db()
    db.execute("INSERT INTO favorite(username, manga_name, created_at) VALUES(?, ?, ?)",(username, manga, created_at))
    db.commit()
    return redirect(f"/product/{manga}")

@app.route("/<manga>/favorite_delete")
@login_required
def favorite_delete(manga):
    username = current_user.username
    db = get_db()
    cur = db.execute(f'DELETE FROM favorite WHERE username = "{username}" AND manga_name = "{manga}"')
    db.commit()
    return redirect(f"/product/{manga}")

@app.route("/hackathon2022",methods=['GET', 'POST'])
def hackathon2022():
    if request.method == 'POST':
        csv_data = request.files['file']
        df = pd.read_csv(csv_data)
        df.to_csv('static/hackathon_image/file.csv', index = False)
        session['df_values'] = df.values.tolist()
        session['df_columns'] = df.columns.tolist()
        session['columns'] = list(df.columns)
        return redirect("/hackathon2022_step2") 
    else:
        session.pop('df_values', None)
        session.pop('df_columns', None)
        session.pop('columns', None)
        session.pop('exceptcolumns', None)
        session.pop('clusters', None)
        session.clear()
        return render_template('hackathon2022.html')

@app.route("/hackathon2022_step2",methods=['GET', 'POST'])
def hackathon2022_step2():
    if request.method == 'POST':
        session['exceptcolumns'] = request.form.getlist('exceptcolumns')
        session['clusters'] = request.form.get('clusters')
        return redirect('/hackathon2022_step3') 
    else:
        df = pd.read_csv('static/hackathon_image/file.csv')
        df_values = df.values.tolist()
        df_columns = df.columns.tolist()
        columns = list(df.columns)
        return render_template('hackathon2022_step2.html', df_values=df_values,df_columns=df_columns, columns=columns)

@app.route("/hackathon2022_step3",methods=['GET', 'POST'])
def hackathon2022_step3():
    if request.method == 'POST':
        df = pd.read_csv('static/hackathon_image/file.csv')
        exceptcolumns = session.get('exceptcolumns', None)
        for i in exceptcolumns:
            df = df.drop(i, axis=1)
        paircolumns = request.form.getlist('paircolumns')
        paircolumns.append('cluster')
        clusters = session.get('clusters', None)
        clusters = int(clusters)

        kmeans = KMeans(n_clusters= clusters, random_state=0, init='random').fit(df)
        df = pd.read_csv('static/hackathon_image/file.csv')
        df['cluster'] = kmeans.labels_
        df = df[paircolumns]
        sns.set()
        sns_plot = sns.pairplot(df, hue='cluster', palette='colorblind', plot_kws={'alpha':0.5})
        fig = sns_plot.savefig('static/hackathon_image/output.png')
        return render_template('hackathon2022_step4.html')
    else:
        df = pd.read_csv('static/hackathon_image/file.csv')
        columns = list(df.columns)
        exceptcolumns = session.get('exceptcolumns', None)
        clusters = session.get('clusters', None)
        clusters = int(clusters)
        for i in exceptcolumns:
            df = df.drop(i, axis=1)

        kmeans = KMeans(n_clusters= clusters, random_state=0, init='random').fit(df)
        df['cluster'] = kmeans.labels_


        pca2 = PCA(n_components=2)
        pca2.fit(df)

        x_pca2 = pca2.transform(df)

        pca_df2 = pd.DataFrame(x_pca2)

        pca_df2['cluster'] = df['cluster'].values

        # 2次元でプロット
        fig = plt.figure(figsize = (8, 8))

        ## 各要素にDataFrameのインデックスの数字をラベルとして付ける
        for i in pca_df2['cluster'].unique():
            tmp = pca_df2.loc[pca_df2['cluster'] == i]
            plt.scatter(tmp[0], tmp[1], label=f'cluster{i}')

        ##凡例を表示 
        plt.legend()


        canvas = FigureCanvasAgg(fig)
        buf = io.BytesIO()
        canvas.print_png(buf)
        data = buf.getvalue()

        img_data = urllib.parse.quote(data)
        return render_template('hackathon2022_step3.html', exceptcolumns=exceptcolumns, clusters=clusters, img_data=img_data, columns=columns)

# @app.route("/test")
# def test():
#     db = get_db()
#     manga = 'SPY×FAMILY'
#     cur = db.execute( f'SELECT round(age/10.0)*10 AS age, COUNT(*) FROM favorite LEFT JOIN user USING(username) WHERE manga_name = "{manga}" AND gender = "男" GROUP BY 1')

#     columns = ['age', 'count']
#     mangas = list(cur)
#     mangas = np.array(mangas)
#     a = np.arange(10,60,10)
#     b = np.zeros(5)
#     c = np.stack([a, b], axis=1)
#     d = np.concatenate([mangas, c])
#     df = pd.DataFrame(d, columns=columns)
#     df['age'] = df['age'].astype(int)
#     df = df.groupby('age').sum()
#     df['count'] = df['count'].astype(int)

#     angles_A = np.linspace(start=0, stop=2*np.pi, num=len(df["count"])+1, endpoint=True)
#     values_A = np.concatenate((df["count"], [df["count"][10]]))

#     fig, ax = plt.subplots(1, 1, figsize=(5, 8), subplot_kw={'projection': 'polar'})
#     ax.fill(angles_A, values_A, alpha=0.25, color="blue")

#     ax.set_thetagrids(angles_A[:-1] * 180 / np.pi, df.index, fontsize=8)
#     ax.set_theta_zero_location('N')

#     ax.set_title("A", fontsize=15)


#     canvas = FigureCanvasAgg(fig)
#     buf = io.BytesIO()
#     canvas.print_png(buf)
#     data = buf.getvalue()

#     img_data = urllib.parse.quote(data)

#     return render_template('test.html', img_data=img_data)

if __name__ == "__main__":
    app.debug = False
    app.run(host='0.0.0.0', port=8888)
