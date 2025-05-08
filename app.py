import pandas as pd
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS, cross_origin
import requests
import json
from tmdbv3api import TMDb
import pickle as pkl
import numpy as np
import random
import bs4
import re
import os
from datetime import datetime, timedelta
import jwt
from functools import wraps

# Import các hàm từ module MySQL
from mysql_integration import (
    add_like_movie, create_connection, get_booked_seats, get_like_movies, get_showtimes, get_user_bookings, save_booking, save_showtime, save_user, get_user_by_firebase_uid, add_to_watchlist, get_watchlist, save_user_preferences, get_user_preferences
)

# Cấu hình TMDB
tmdb = TMDb()
tmdb.api_key = 'dbda4bb34573ea2b68379f1e476c3933'
from tmdbv3api import Movie
tmdb_movie = Movie()

# Đọc dữ liệu phim
df2 = pd.read_csv("Main_data.csv")

# Đọc các model đã train
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
vectorizer = pkl.load(open('vectorizerer.pkl', 'rb'))
clt = pkl.load(open('nlp_model.pkl', 'rb'))

# URL cho các API call
url = [
    "http://api.themoviedb.org/3/discover/movie?api_key=dbda4bb34573ea2b68379f1e476c3933&primary_release_year=2015&adult=false",
    "http://api.themoviedb.org/3/discover/movie?api_key=dbda4bb34573ea2b68379f1e476c3933&primary_release_year=2014&adult=false",
    "https://api.themoviedb.org/3/movie/popular?api_key=dbda4bb34573ea2b68379f1e476c3933&language=en-US&page=1&adult=false",
    "https://api.themoviedb.org/3/movie/popular?api_key=dbda4bb34573ea2b68379f1e476c3933&language=en-US&page=2&adult=false",
    "https://api.themoviedb.org/3/movie/popular?api_key=dbda4bb34573ea2b68379f1e476c3933&language=en-US&page=3&adult=false",
    "https://api.themoviedb.org/3/discover/movie?api_key=dbda4bb34573ea2b68379f1e476c3933&with_genres=18&adult=false",
    "http://api.themoviedb.org/3/discover/movie?api_key=dbda4bb34573ea2b68379f1e476c3933&primary_release_year=2020&adult=false",
    "http://api.themoviedb.org/3/discover/movie?api_key=dbda4bb34573ea2b68379f1e476c3933&primary_release_year=2019&adult=false",
    "http://api.themoviedb.org/3/discover/movie?api_key=dbda4bb34573ea2b68379f1e476c3933&primary_release_year=2017&adult=false",
    "http://api.themoviedb.org/3/discover/movie?api_key=dbda4bb34573ea2b68379f1e476c3933&primary_release_year=2016&adult=false",
    "https://api.themoviedb.org/3/discover/movie?api_key=dbda4bb34573ea2b68379f1e476c3933&with_genres=27",
    "https://api.themoviedb.org/3/discover/movie?api_key=dbda4bb34573ea2b68379f1e476c3933&with_genres=16"
]

# Các hàm đã có trong mã gốc giữ nguyên
def get_news():
    response = requests.get("https://www.imdb.com/news/top/?ref_=hm_nw_sm")
    soup = bs4.BeautifulSoup(response.text, 'html.parser')
    data = [re.sub('[\n()]', "", d.text) for d in soup.find_all('div', class_='news-article__content')]
    image = [m['src'] for m in soup.find_all("img", class_="news-article__image")]
    t_data = []
    for i in range(len(data)):
        t_data.append([image[i], data[i][1:len(data[i])-1]])
    return t_data

def getdirector(x):
    data = []
    result = tmdb_movie.search(x)
    movie_id = result[0].id
    response = requests.get(
        "https://api.themoviedb.org/3/movie/{}/credits?api_key=dbda4bb34573ea2b68379f1e476c3933".format(
            movie_id))
    data_json = response.json()
    data.append(data_json)
    crew=data[0]['crew']

    director=[]
    for c in crew:
        if c['job']=='Director':
            director.append(c['name'])
            break
    return director

def get_swipe():
    data=[]
    val=random.choice(url)
    for i in range(5):
        lis=[]
        response = requests.get(
            val+"&page="+str(i+1))
        data_json = response.json()
        lis.append(data_json["results"])
        for i in lis[0]:
            data.append(i)
    return data

def getreview(x):
    data=[]
    result=tmdb_movie.search(x)
    movie_id=result[0].id
    response=requests.get("https://api.themoviedb.org/3/movie/{}/reviews?api_key=dbda4bb34573ea2b68379f1e476c3933&language=en-US&page=1".format(movie_id))
    data_json=response.json()
    data.append(data_json)
    return data

def getrating(title):
    movie_review = []
    data=getreview(title)
    for i in data[0]['results']:
        pred=clt.predict(vectorizer.transform([i['content']]))
        if pred[0]=='positive':
            movie_review.append({
                "review":i['content'],
                "rating":"Good"
            })
        else:
            movie_review.append({
                "review": i['content'],
                "rating": "Bad"
            })
    return movie_review

def get_data(x):
    data=[]
    result=tmdb_movie.search(x)
    movie_id=result[0].id
    response=requests.get("https://api.themoviedb.org/3/movie/{}?api_key=dbda4bb34573ea2b68379f1e476c3933".format(movie_id))
    response2=requests.get("https://api.themoviedb.org/3/movie/{}/credits?api_key=dbda4bb34573ea2b68379f1e476c3933".format(movie_id))
    response3=requests.get("https://api.themoviedb.org/3/movie/{}/keywords?api_key=dbda4bb34573ea2b68379f1e476c3933".format(movie_id))
    data_json=response.json()
    data_json2=response2.json()
    data_json3=response3.json()
    data.append(data_json)
    data.append(data_json2)
    data.append(data_json3)
    return data

def getcomb(movie_data):
    cast_data=movie_data[1]['cast']
    cast=[]
    for data in cast_data:
        cast.append(data['name'])
    crew=movie_data[1]['crew']
    director=[]
    for c in crew:
        if c['job']=='Director':
            director.append(c['name'])
            break
    genres=[]
    for x in movie_data[0]['genres']:
        genres.append(x['name'])
    keywords=[]
    for k in movie_data[2]['keywords']:
        keywords.append(k['name'])
    d=str(cast)+str(keywords)+str(genres)+director[0]+str(movie_data[0]['overview'])
    return d

def get_recommendations(title):
    movie_data=get_data(title)
    total_data=getcomb(movie_data)
    df2=pd.read_csv("Main_data.csv")
    df2['comb']=df2['cast']+df2['director']+df2['genres']+df2['keywords']+df2['overview']
    myseries=pd.Series(data=[total_data,title],index=['comb','title_x'])
    flag=0
    for i in df2['title_x']:
        if i==title:
            flag=1
    if flag==0:
        df2 = pd.concat([df2, myseries.to_frame().T], ignore_index=True)
    df2=df2.replace(np.nan,'')
    tfidf = TfidfVectorizer(stop_words='english')
    count_matrix = tfidf.fit_transform(df2['comb'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    indices = pd.Series(df2.index, index=df2['title_x'])
    idx=indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:10]
    movie_indices = [i[0] for i in sim_scores]
    return df2['title_x'].iloc[movie_indices]

def get_data2(x):
    data=[]
    result=tmdb_movie.search(x)
    movie_id=result[0].id
    trailer=requests.get("https://api.themoviedb.org/3/movie/{}/videos?api_key=dbda4bb34573ea2b68379f1e476c3933&language=en-US".format(movie_id))
    response=requests.get("https://api.themoviedb.org/3/movie/{}?api_key=dbda4bb34573ea2b68379f1e476c3933".format(movie_id))
    data_json = response.json()
    trailer=trailer.json()
    data.append(data_json)
    data.append(trailer)
    return data

# FLASK
app = Flask(__name__)
cors = CORS(app)

# Cấu hình secret key cho session và token
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)

# Decorator để xác thực token
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Lấy token từ header
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
        
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        try:
            # Giải mã token
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = get_user_by_firebase_uid(data['firebase_uid'])
            if not current_user:
                return jsonify({'message': 'User not found!'}), 401
        except Exception as e:
            return jsonify({'message': 'Token is invalid!', 'error': str(e)}), 401
        
        return f(current_user, *args, **kwargs)
    
    return decorated

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    
    if not data or not data.get('firebase_uid') or not data.get('email') or not data.get('name'):
        return jsonify({'message': 'Missing registration data!'}), 400
    
    try:
        firebase_uid = data.get('firebase_uid')
        email = data.get('email')
        display_name = data.get('display_name') or data.get('name')
        photo_url = data.get('photo_url')
        
        user_id = save_user(firebase_uid, email, display_name, photo_url)
        
        if not user_id:
            return jsonify({'message': 'Failed to save user!'}), 500
        
        # Tạo JWT token
        token = jwt.encode({
            'firebase_uid': firebase_uid,
            'email': email,
            'exp': datetime.utcnow() + timedelta(days=30)
        }, app.config['SECRET_KEY'], algorithm="HS256")
        
        return jsonify({
            'token': token,
            'user_id': user_id,
            'email': email,
            'name': display_name  # Client sử dụng tên này
        })
        
    except Exception as e:
        return jsonify({'message': f'Registration failed: {str(e)}'}), 400
    
@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    
    if not data or not data.get('firebase_uid') or not data.get('email'):
        return jsonify({'message': 'Missing login data!'}), 400
    
    firebase_uid = data.get('firebase_uid')
    email = data.get('email')
    display_name = data.get('display_name')
    photo_url = data.get('photo_url')
    
    # Lưu thông tin người dùng vào MySQL
    user_id = save_user(firebase_uid, email, display_name, photo_url)
    
    if not user_id:
        return jsonify({'message': 'Failed to save user!'}), 500
    
    # Tạo token JWT
    token = jwt.encode({
        'firebase_uid': firebase_uid,
        'email': email,
        'exp': datetime.utcnow() + timedelta(days=30)
    }, app.config['SECRET_KEY'], algorithm="HS256")
    
    return jsonify({
        'token': token,
        'user_id': user_id,
        'email': email,
        'name': display_name  # Thống nhất response với route register
    })

@app.route('/api/user/profile', methods=['GET'])
@token_required
def get_user_profile(current_user):
    # Thông tin người dùng đã được lấy từ token_required decorator
    return jsonify({
        'id': current_user['id'],
        'email': current_user['email'],
        'display_name': current_user['display_name'],
        'photo_url': current_user['photo_url'],
        'created_at': current_user['created_at'],
        'last_login': current_user['last_login']
    })

@app.route('/api/user/watchlist', methods=['GET'])
@token_required
def user_watchlist(current_user):
    # Lấy danh sách phim muốn xem của người dùng
    watchlist = get_watchlist(current_user['id'])
    return jsonify(watchlist)

@app.route('/api/user/watchlist/sync', methods=['POST'])
@token_required
def sync_watchlist(current_user):
    data = request.get_json()
    
    if not data or 'watchlist' not in data:
        return jsonify({'message': 'Missing watchlist data!'}), 400
    
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            
            # Xóa danh sách cũ để đồng bộ mới
            cursor.execute("DELETE FROM watchlist WHERE user_id = %s", (current_user['id'],))
            
            # Thêm các phim mới từ danh sách
            for item in data['watchlist']:
                if 'movie_id' in item and 'movie_title' in item:
                    poster_path = item.get('poster_path', '')
                    cursor.execute("""
                        INSERT INTO watchlist (user_id, movie_id, movie_title, poster_path)
                        VALUES (%s, %s, %s, %s)
                    """, (current_user['id'], item['movie_id'], item['movie_title'], poster_path))
            
            connection.commit()
            return jsonify({'message': 'Watchlist synced successfully'})
        except Error as e:
            print(f"Error syncing watchlist: {e}")
            connection.rollback()
            return jsonify({'message': f'Failed to sync watchlist: {str(e)}'}), 500
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    return jsonify({"message": "Database connection error"}), 500

# @app.route('/api/check-token', methods=['GET'])
# @token_required
# def check_token(current_user):
#     # This endpoint just validates that the token is still valid
#     return jsonify({'valid': True, 'user_id': current_user['id']})

@app.route('/api/auth/verify', methods=['GET'])
@token_required
def verify_token(current_user):
    return jsonify({
        'valid': True,
        'user_id': current_user['id'],
        'email': current_user['email'],
        'name': current_user['display_name']
    })

@app.route('/api/user/like', methods=['GET'])
@token_required
def user_watched(current_user):
    # Lấy danh sách phim đã xem
    watched = get_like_movies(current_user['id'])
    return jsonify(watched)

@app.route('/api/user/like/add', methods=['POST'])
@token_required
def add_user_watched(current_user):
    data = request.get_json()
    
    if not data or 'movie_id' not in data or 'movie_title' not in data:
        return jsonify({'message': 'Missing movie information!'}), 400
    
    result = add_like_movie(
        current_user['id'], 
        data['movie_id'], 
        data['movie_title'],
        data['poster_path'],
    )
    
    if result:
        return jsonify({'message': 'Movie added to watched list successfully'})
    else:
        return jsonify({'message': 'Failed to add movie to watched list'}), 500
    
@app.route('/api/user/watchlist/add', methods=['POST'])
@token_required
def add_user_watchlist(current_user):
    data = request.get_json()
    
    if not data or 'movie_id' not in data or 'movie_title' not in data:
        return jsonify({'message': 'Missing movie information!'}), 400
    
    # Thêm phim vào danh sách đã xem
    result = add_to_watchlist(
        current_user['id'], 
        data['movie_id'], 
        data['movie_title'],
        data['poster_path']
    )
    
    if result:
        return jsonify({'message': 'Movie added to watched list successfully'})
    else:
        return jsonify({'message': 'Failed to add movie to watched list'}), 500

@app.route('/api/user/preferences', methods=['GET'])
@token_required
def user_preferences(current_user):
    # Lấy sở thích thể loại phim
    preferences = get_user_preferences(current_user['id'])
    return jsonify(preferences)

@app.route('/api/user/preferences/save', methods=['POST'])
@token_required
def save_preferences(current_user):
    data = request.get_json()
    
    if not data:
        return jsonify({'message': 'Missing data!'}), 400
    
    # Kiểm tra và lấy thể loại từ request
    genres = data.get('genres', [])
    languages = data.get('languages', [])
    
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            
            # Xóa preferences cũ
            cursor.execute("DELETE FROM user_preferences WHERE user_id = %s", (current_user['id'],))
            
            # Thêm preferences thể loại mới
            for genre in genres:
                cursor.execute("""
                    INSERT INTO user_preferences (user_id, preference_type, preference_value)
                    VALUES (%s, 'genre', %s)
                """, (current_user['id'], genre))
            
            # Thêm preferences ngôn ngữ mới
            for language in languages:
                cursor.execute("""
                    INSERT INTO user_preferences (user_id, preference_type, preference_value)
                    VALUES (%s, 'language', %s)
                """, (current_user['id'], language))
            
            connection.commit()
            return jsonify({'message': 'Preferences saved successfully'})
        except Error as e:
            print(f"Error saving preferences: {e}")
            connection.rollback()
            return jsonify({'message': f'Failed to save preferences: {str(e)}'}), 500
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    return jsonify({"message": "Database connection error"}), 500

# API endpoint đề xuất phim dựa trên sở thích của người dùng
@app.route('/api/user/recommendations', methods=['GET'])
@token_required
def user_recommendations(current_user):
    # Lấy sở thích thể loại phim
    preferences = get_user_preferences(current_user['id'])
    
    # Lấy danh sách phim đã xem
    watched = get_like_movies(current_user['id'])
    watched_ids = [movie['movie_id'] for movie in watched]
    
    # Lấy phim dựa trên sở thích
    recommendations = []
    
    # Kiểm tra preferences và chuyển đổi sang list nếu cần
    genre_list = preferences['genres'] if isinstance(preferences, dict) and 'genres' in preferences else []
    if isinstance(preferences, list):
        genre_list = preferences
    
    # Giới hạn số lượng thể loại
    genre_count = min(3, len(genre_list))
    
    # Nếu có sở thích thể loại
    for i in range(genre_count):
        genre = genre_list[i]
        for page in range(1, 3):  # Lấy 2 trang đầu tiên
            try:
                response = requests.get(
                    f"https://api.themoviedb.org/3/discover/movie?api_key=dbda4bb34573ea2b68379f1e476c3933&with_genres={genre}&page={page}"
                )
                data = response.json()
                
                # Lọc phim chưa xem
                for movie in data.get('results', []):
                    if movie['id'] not in watched_ids and movie not in recommendations:
                        recommendations.append(movie)
                        
                        # Giới hạn số phim đề xuất
                        if len(recommendations) >= 20:
                            break
            except Exception as e:
                print(f"Error fetching recommendations for genre {genre}: {e}")
                
    # Nếu không có đủ đề xuất từ sở thích, thêm phim phổ biến
    if len(recommendations) < 20:
        try:
            response = requests.get(
                "https://api.themoviedb.org/3/movie/popular?api_key=dbda4bb34573ea2b68379f1e476c3933&language=en-US&page=1"
            )
            data = response.json()
            
            # Lọc phim chưa xem và chưa có trong danh sách đề xuất
            for movie in data.get('results', []):
                if movie['id'] not in watched_ids and movie not in recommendations:
                    recommendations.append(movie)
                
                # Giới hạn số phim đề xuất
                if len(recommendations) >= 20:
                    break
        except Exception as e:
            print(f"Error fetching popular movies: {e}")
    
    return jsonify(recommendations[:20])  # Giới hạn 20 phim

# API để lấy đề xuất dựa trên phim đã xem gần đây
@app.route('/api/user/recommendations/based-on-watched', methods=['GET'])
@token_required
def recommendations_based_on_watched(current_user):
    # Lấy danh sách phim đã xem
    watched = get_like_movies(current_user['id'])
    
    if not watched:
        return jsonify([])  # Trả về rỗng nếu chưa xem phim nào
    
    # Lấy phim đã xem gần nhất
    latest_watched = watched[0]  # Đã được sắp xếp theo thời gian giảm dần
    
    try:
        # Sử dụng hàm get_recommendations đã có để lấy đề xuất
        recommended_titles = get_recommendations(latest_watched['movie_title'])
        result = []
        
        # Lấy thông tin chi tiết cho mỗi phim được đề xuất
        for title in recommended_titles:
            movie_data = get_data2(title)
            result.append(movie_data[0])
        
        return jsonify(result)
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return jsonify([]), 500

# API endpoint đã có giữ nguyên
@app.route('/getname', methods=["GET"])
def getnames():
    data = []
    for i in df2["title_x"]:
        data.append(i)
    return jsonify(data)

@app.route('/getmovie/<movie_name>', methods=["GET"])
def getmovie(movie_name):
    data = get_data2(movie_name)
    return jsonify(data)

@app.route('/getreview/<movie_name>', methods=["GET"])
def getreviews(movie_name):
    data = getrating(movie_name)
    return jsonify(data)

@app.route('/getdirector/<movie_name>', methods=["GET"])
def getdirectorname(movie_name):
    data = getdirector(movie_name)
    return jsonify(data)

@app.route('/getswipe', methods=["GET"])
def getswipe():
    data = get_swipe()
    return jsonify(data)

@app.route('/getnews', methods=["GET"])
def getnewsdata():
    data = get_news()
    return jsonify(data)

@app.route('/send/<movie_name>', methods=["GET"])
def get(movie_name):
    if request.method == "GET":
        val = get_recommendations(movie_name)
        if val is None:
            return jsonify({"message": "movie not found in database"})
        val = list(val)
        result = []
        try:
            for i in val:
                res = get_data2(i)
                result.append(res[0])
        except requests.ConnectionError:
            return jsonify({"message": "movie not found in database"})
        return jsonify(result)

# API để tìm kiếm phim theo từ khóa
@app.route('/api/search', methods=['GET'])
def search_movies():
    query = request.args.get('query', '')
    if not query:
        return jsonify([])
    
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/search/movie?api_key=dbda4bb34573ea2b68379f1e476c3933&language=en-US&query={query}&page=1&include_adult=false"
        )
        data = response.json()
        return jsonify(data.get('results', []))
    except Exception as e:
        print(f"Error searching movies: {e}")
        return jsonify([]), 500

# API để lấy thông tin chi tiết của phim bằng ID
@app.route('/api/movie/<int:movie_id>', methods=['GET'])
def get_movie_by_id(movie_id):
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=dbda4bb34573ea2b68379f1e476c3933&append_to_response=videos,credits,recommendations"
        )
        data = response.json()
        return jsonify(data)
    except Exception as e:
        print(f"Error getting movie by ID: {e}")
        return jsonify({"message": "Movie not found"}), 404

# API để xóa phim khỏi watchlist
@app.route('/api/user/watchlist/remove/<int:movie_id>', methods=['DELETE'])
@token_required
def remove_from_watchlist(current_user, movie_id):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("""
                DELETE FROM watchlist 
                WHERE user_id = %s AND movie_id = %s
            """, (current_user['id'], movie_id))
            
            connection.commit()
            return jsonify({"message": "Movie removed from watchlist"})
        except Error as e:
            print(f"Error removing from watchlist: {e}")
            connection.rollback()
            return jsonify({"message": "Failed to remove movie from watchlist"}), 500
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    return jsonify({"message": "Database connection error"}), 500

@app.route('/api/user/like/remove/<int:movie_id>', methods=['DELETE'])
@token_required
def remove_from_like(current_user, movie_id):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("""
                DELETE FROM likes 
                WHERE user_id = %s AND movie_id = %s
            """, (current_user['id'], movie_id))
            
            connection.commit()
            return jsonify({"message": "Movie removed from watchlist"})
        except Error as e:
            print(f"Error removing from watchlist: {e}")
            connection.rollback()
            return jsonify({"message": "Failed to remove movie from watchlist"}), 500
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    return jsonify({"message": "Database connection error"}), 500

# API để lấy các thể loại phim
@app.route('/api/genres', methods=['GET'])
def get_genres():
    try:
        response = requests.get(
            "https://api.themoviedb.org/3/genre/movie/list?api_key=dbda4bb34573ea2b68379f1e476c3933&language=en-US"
        )
        data = response.json()
        return jsonify(data.get('genres', []))
    except Exception as e:
        print(f"Error getting genres: {e}")
        return jsonify([]), 500
    

@app.route('/api/movies/<int:movie_id>/showtimes', methods=['GET'])
def get_movie_showtimes(movie_id):
    show_date = request.args.get('date')
    if not show_date:
        return jsonify({'message': 'Date is required'}), 400
    
    try:
        showtimes = get_showtimes(str(movie_id), show_date)
        return jsonify({'showtimes': showtimes})
    except Exception as e:
        print(f"Error getting showtimes: {e}")
        return jsonify({'message': f'Failed to get showtimes: {str(e)}'}), 500
    
@app.route('/api/showtimes/create', methods=['POST'])
@token_required
def create_showtime(current_user):
    data = request.get_json()
    
    if not data or not all(k in data for k in ['movie_id', 'show_date', 'show_time', 'theater_name']):
        return jsonify({'message': 'Missing showtime information!'}), 400
    
    showtime_id = save_showtime(
        movie_id=data['movie_id'],
        show_date=data['show_date'],
        show_time=data['show_time'],
        theater_name=data['theater_name']
    )
    
    if showtime_id:
        return jsonify({'message': 'Showtime created successfully', 'showtime_id': showtime_id})
    else:
        return jsonify({'message': 'Failed to create showtime'}), 500

@app.route('/api/showtimes/<int:showtime_id>/seats', methods=['GET'])
def get_seat_availability(showtime_id):
    try:
        booked_seats = get_booked_seats(showtime_id)
        return jsonify({'booked_seats': booked_seats})
    except Exception as e:
        print(f"Error getting seat availability: {e}")
        return jsonify({'message': f'Failed to get seat availability: {str(e)}'}), 500

@app.route('/api/bookings', methods=['POST'])
@token_required
def create_booking(current_user):
    data = request.get_json()
    
    if not data or not all(k in data for k in ['showtime_id', 'seat_ids', 'total_price', 'payment_method']):
        return jsonify({'message': 'Missing booking information!'}), 400
    
    showtime_id = data['showtime_id']
    seat_ids = data['seat_ids']
    total_price = data['total_price']
    payment_method = data['payment_method']
    
    # Kiểm tra xem ghế đã được đặt chưa
    booked_seats = get_booked_seats(showtime_id)
    if any(seat in booked_seats for seat in seat_ids):
        return jsonify({'message': 'One or more seats are already booked'}), 400
    
    booking_id = save_booking(
        user_id=current_user['id'],
        showtime_id=showtime_id,
        seat_ids=seat_ids,
        total_price=total_price,
        payment_method=payment_method
    )
    
    if booking_id:
        return jsonify({
            'message': 'Booking created successfully',
            'booking_id': booking_id
        })
    else:
        return jsonify({'message': 'Failed to create booking'}), 500

@app.route('/api/user/bookings', methods=['GET'])
@token_required
def get_user_bookings_route(current_user):
    try:
        bookings = get_user_bookings(current_user['id'])
        return jsonify(bookings)
    except Exception as e:
        print(f"Error getting user bookings: {e}")
        return jsonify({'message': f'Failed to get bookings: {str(e)}'}), 500
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')