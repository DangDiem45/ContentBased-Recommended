import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from flask import Flask
from app import app
from mysql_integration import (
    save_user, get_user_by_firebase_uid, add_to_watchlist, get_watchlist,
    add_like_movie, get_like_movies, save_user_preferences, get_user_preferences,
    save_showtime, get_showtimes, save_booking, get_booked_seats, get_user_bookings,
    create_connection
)
from models import TfidfVectorizerManual, LogisticRegressionManual, cosine_similarity_manual
import json
import jwt
import requests_mock
from mysql.connector import connect, Error

# Suppress Werkzeug and ast deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="werkzeug")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ast")

# Fixture để tạo ứng dụng Flask
@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['SECRET_KEY'] = 'test-secret-key'
    with app.test_client() as client:
        yield client

# Fixture để thiết lập và dọn dẹp cơ sở dữ liệu
@pytest.fixture
def mysql_db():
    # Thông tin kết nối tới database movie_film
    connection = connect(
        host="localhost",
        user="root",  # Sử dụng user 'root' để khớp với mysql_integration.py
        password="",  # Không có mật khẩu, để trống để khớp với mysql_integration.py
        database="movie_film"
    )
    
    cursor = connection.cursor()
    
    # Dọn dẹp dữ liệu trước khi chạy test
    cursor.execute("DELETE FROM bookings")
    cursor.execute("DELETE FROM showtimes")
    cursor.execute("DELETE FROM likes")
    cursor.execute("DELETE FROM watchlist")
    cursor.execute("DELETE FROM user_preferences")
    cursor.execute("DELETE FROM users")
    connection.commit()
    
    yield connection
    
    # Dọn dẹp dữ liệu sau khi test
    cursor.execute("DELETE FROM bookings")
    cursor.execute("DELETE FROM showtimes")
    cursor.execute("DELETE FROM likes")
    cursor.execute("DELETE FROM watchlist")
    cursor.execute("DELETE FROM user_preferences")
    cursor.execute("DELETE FROM users")
    connection.commit()
    
    cursor.close()
    connection.close()

# Fixture để tạo token JWT
@pytest.fixture
def auth_token():
    token = jwt.encode({
        'firebase_uid': 'test_user_123',
        'email': 'test@example.com',
        'exp': datetime.now(timezone.utc) + timedelta(days=30)
    }, app.config['SECRET_KEY'], algorithm="HS256")
    return token

# --- Test MySQL Integration Functions ---

def test_save_user(mysql_db):
    user_id = save_user(
        firebase_uid='test_user_123',
        email='test@example.com',
        display_name='Test User',
        photo_url='http://example.com/photo.jpg'
    )
    assert user_id is not None
    user = get_user_by_firebase_uid('test_user_123')
    assert user['email'] == 'test@example.com'
    assert user['display_name'] == 'Test User'

def test_add_to_watchlist(mysql_db):
    user_id = save_user('test_user_123', 'test@example.com')
    result = add_to_watchlist(user_id, '12345', 'Test Movie', 'http://example.com/poster.jpg')
    assert result is True
    watchlist = get_watchlist(user_id)
    assert len(watchlist) == 1
    assert watchlist[0]['movie_title'] == 'Test Movie'

def test_add_like_movie(mysql_db):
    user_id = save_user('test_user_123', 'test@example.com')
    add_to_watchlist(user_id, '12345', 'Test Movie', 'http://example.com/poster.jpg')
    result = add_like_movie(user_id, '12345', 'Test Movie', 'http://example.com/poster.jpg')
    assert result is True
    likes = get_like_movies(user_id)
    assert len(likes) == 1
    assert likes[0]['movie_title'] == 'Test Movie'
    watchlist = get_watchlist(user_id)
    assert len(watchlist) == 0  # Phim đã được xóa khỏi watchlist

def test_save_user_preferences(mysql_db):
    user_id = save_user('test_user_123', 'test@example.com')
    result = save_user_preferences(user_id, genres=['Action', 'Drama'], languages=['en', 'fr'])
    assert result is True
    preferences = get_user_preferences(user_id)
    assert set(preferences['genres']) == {'Action', 'Drama'}
    assert set(preferences['languages']) == {'en', 'fr'}

def test_save_showtime(mysql_db):
    showtime_id = save_showtime(
        movie_id='12345',
        show_date='2025-05-28',
        show_time='18:00:00',
        theater_name='Cinema 1'
    )
    assert showtime_id is not None
    showtimes = get_showtimes('12345', '2025-05-28')
    assert len(showtimes) == 1
    assert showtimes[0]['theater_name'] == 'Cinema 1'

def test_save_booking(mysql_db):
    user_id = save_user('test_user_123', 'test@example.com')
    showtime_id = save_showtime('12345', '2025-05-28', '18:00:00', 'Cinema 1')
    booking_id = save_booking(
        user_id=user_id,
        showtime_id=showtime_id,
        seat_ids=['A1', 'A2'],
        total_price=20.00,
        payment_method='credit_card'
    )
    assert booking_id is not None
    bookings = get_user_bookings(user_id)
    assert len(bookings) == 1
    assert bookings[0]['total_price'] == 20.00
    assert bookings[0]['seat_ids'] == ['A1', 'A2']

def test_get_booked_seats(mysql_db):
    user_id = save_user('test_user_123', 'test@example.com')
    showtime_id = save_showtime('12345', '2025-05-28', '18:00:00', 'Cinema 1')
    save_booking(user_id, showtime_id, ['A1', 'A2'], 20.00, 'credit_card')
    booked_seats = get_booked_seats(showtime_id)
    assert set(booked_seats) == {'A1', 'A2'}

# --- Test Flask Endpoints ---

def test_register(client, mysql_db):
    data = {
        'firebase_uid': 'test_user_123',
        'email': 'test@example.com',
        'name': 'Test User',
        'photo_url': 'http://example.com/photo.jpg'
    }
    response = client.post('/api/auth/register', json=data)
    assert response.status_code == 200
    assert response.json['email'] == 'test@example.com'
    assert response.json['name'] == 'Test User'
    assert 'token' in response.json

def test_login(client, mysql_db):
    save_user('test_user_123', 'test@example.com', 'Test User', 'http://example.com/photo.jpg')
    data = {
        'firebase_uid': 'test_user_123',
        'email': 'test@example.com',
        'display_name': 'Test User',
        'photo_url': 'http://example.com/photo.jpg'
    }
    response = client.post('/api/auth/login', json=data)
    assert response.status_code == 200
    assert response.json['email'] == 'test@example.com'
    assert response.json['name'] == 'Test User'
    assert 'token' in response.json

def test_user_watchlist(client, auth_token, mysql_db):
    user_id = save_user('test_user_123', 'test@example.com')
    add_to_watchlist(user_id, '12345', 'Test Movie', 'http://example.com/poster.jpg')
    response = client.get('/api/user/watchlist', headers={'Authorization': f'Bearer {auth_token}'})
    assert response.status_code == 200
    assert len(response.json) == 1
    assert response.json[0]['movie_title'] == 'Test Movie'

def test_add_to_watchlist_endpoint(client, auth_token, mysql_db):
    save_user('test_user_123', 'test@example.com')
    data = {
        'movie_id': '12345',
        'movie_title': 'Test Movie',
        'poster_path': 'http://example.com/poster.jpg'
    }
    response = client.post('/api/user/watchlist/add', json=data,
                         headers={'Authorization': f'Bearer {auth_token}'})
    assert response.status_code == 200
    assert response.json['message'] == 'Movie added to watched list successfully'

def test_user_recommendations(client, auth_token, mysql_db, requests_mock):
    # Lưu user và lấy user_id thực tế
    user_id = save_user('test_user_123', 'test@example.com')
    # Sử dụng user_id thực tế thay vì hardcode user_id=1
    save_user_preferences(user_id=user_id, genres=['Action'], languages=['en'])
    
    # Mock API call cho discover/movie
    requests_mock.get(
        'https://api.themoviedb.org/3/discover/movie',
        json={'results': [{'id': 1, 'title': 'Mock Movie', 'poster_path': '/mock.jpg'}]}
    )
    # Thêm mock cho movie/popular
    requests_mock.get(
        'https://api.themoviedb.org/3/movie/popular',
        json={'results': [{'id': 2, 'title': 'Popular Movie', 'poster_path': '/popular.jpg'}]}
    )
    
    response = client.get('/api/user/recommendations', headers={'Authorization': f'Bearer {auth_token}'})
    assert response.status_code == 200
    assert len(response.json) > 0
    assert response.json[0]['title'] == 'Mock Movie'

def test_search_movies(client, requests_mock):
    requests_mock.get(
        'https://api.themoviedb.org/3/search/movie',
        json={'results': [{'id': 1, 'title': 'Search Result', 'poster_path': '/search.jpg'}]}
    )
    response = client.get('/api/search?query=Test')
    assert response.status_code == 200
    assert len(response.json) > 0
    assert response.json[0]['title'] == 'Search Result'

def test_create_booking(client, auth_token, mysql_db):
    user_id = save_user('test_user_123', 'test@example.com')
    showtime_id = save_showtime('12345', '2025-05-28', '18:00:00', 'Cinema 1')
    data = {
        'showtime_id': showtime_id,
        'seat_ids': ['A1', 'A2'],
        'total_price': 20.00,
        'payment_method': 'credit_card'
    }
    response = client.post('/api/bookings', json=data,
                         headers={'Authorization': f'Bearer {auth_token}'})
    assert response.status_code == 200
    assert response.json['message'] == 'Booking created successfully'

# --- Test Model Functions ---

def test_tfidf_vectorizer():
    vectorizer = TfidfVectorizerManual(max_features=5, stop_words=['the'])
    texts = [
        "this is a test document",
        "another test document with more words"
    ]
    X = vectorizer.fit_transform(texts)
    assert X.shape == (2, 5)  # 2 tài liệu, 5 từ vựng
    assert len(vectorizer.vocab) <= 5
    X_transform = vectorizer.transform(["test document"])
    assert X_transform.shape == (1, 5)

def test_logistic_regression():
    model = LogisticRegressionManual(learning_rate=0.1, max_iter=10)
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == 4
    assert all(pred in [0, 1] for pred in predictions)

def test_cosine_similarity():
    X = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    sim_matrix = cosine_similarity_manual(X)
    assert sim_matrix.shape == (3, 3)
    assert np.allclose(sim_matrix.diagonal(), 1.0)  # Đường chéo là 1
    assert np.allclose(sim_matrix[0, 1], 0.0)  # Các vector vuông góc