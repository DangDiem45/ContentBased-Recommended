import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import requests
from app import app
from mysql_integration import (
    save_user, get_user_by_firebase_uid, add_to_watchlist, get_watchlist,
    add_like_movie, get_like_movies, save_user_preferences, get_user_preferences,
    save_showtime, get_showtimes, save_booking, get_booked_seats, get_user_bookings, create_connection
)
from models import TfidfVectorizerManual, LogisticRegressionManual, cosine_similarity_manual
import jwt
from mysql.connector import connect

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
    connection = connect(
        host="localhost",
        user="root", 
        password="",  
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

def test_save_user():
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

def test_add_to_watchlist():
    user_id = save_user('test_user_123', 'test@example.com')
    result = add_to_watchlist(user_id, '12345', 'Test Movie', 'http://example.com/poster.jpg')
    assert result is True
    watchlist = get_watchlist(user_id)
    assert len(watchlist) == 1
    assert watchlist[0]['movie_title'] == 'Test Movie'

def test_add_like_movie():
    user_id = save_user('test_user_123', 'test@example.com')
    add_to_watchlist(user_id, '12345', 'Test Movie', 'http://example.com/poster.jpg')
    result = add_like_movie(user_id, '12345', 'Test Movie', 'http://example.com/poster.jpg')
    assert result is True
    likes = get_like_movies(user_id)
    assert len(likes) == 1
    assert likes[0]['movie_title'] == 'Test Movie'
    watchlist = get_watchlist(user_id)
    assert len(watchlist) == 0 

def test_save_user_preferences(mysql_db):
    user_id = save_user('test_user_123', 'test@example.com')
    result = save_user_preferences(user_id, genres=['Action', 'Drama'], languages=['en', 'fr'])
    assert result is True
    preferences = get_user_preferences(user_id)
    assert set(preferences['genres']) == {'Action', 'Drama'}
    assert set(preferences['languages']) == {'en', 'fr'}

def test_save_showtime():
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

def test_save_booking():
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



def test_register(client):
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
    assert X.shape == (2, 5)
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

def test_index_route(client):
    """Test the index route returns a template."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"<title>Movie Api</title>" in response.data  # Kiểm tra title trong HTML

def test_get_movie_route_success(client, mocker):
    """Test the /getmovie route with a valid movie name."""
    mock_data = {"title": "Test Movie", "id": 123}
    mocker.patch('app.get_data2', return_value=[mock_data, {"results": [{"key": "trailer_key"}]}])
    response = client.get('/getmovie/Test Movie')
    assert response.status_code == 200
    data = response.get_json()
    assert data[0]["title"] == "Test Movie"

def test_get_reviews_route_success(client, mocker):
    """Test the /getreview route with a valid movie name."""
    mock_reviews = [{"review": "Good review", "rating": "Good"}]
    mocker.patch('app.getrating', return_value=mock_reviews)
    response = client.get('/getreview/Test Movie')
    assert response.status_code == 200
    data = response.get_json()
    assert data == mock_reviews

def test_get_director_route_success(client, mocker):
    """Test the /getdirector route with a valid movie name."""
    mocker.patch('app.getdirector', return_value=["Test Director"])
    response = client.get('/getdirector/Test Movie')
    assert response.status_code == 200
    data = response.get_json()
    assert data == ["Test Director"]

def test_get_swipe_route(client, mocker):
    """Test the /getswipe route returns movie data."""
    mock_data = [{"title": f"Movie {i}", "id": i} for i in range(5)]
    mocker.patch('app.get_swipe', return_value=mock_data)
    response = client.get('/getswipe')
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 5
    assert all("title" in movie for movie in data)

def test_get_news_route(client, mocker):
    """Test the /getnews route returns news data."""
    mock_news = [["image1.jpg", "News 1"], ["image2.jpg", "News 2"]]
    mocker.patch('app.get_news', return_value=mock_news)
    response = client.get('/getnews')
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 2
    assert all(len(item) == 2 for item in data)

def test_send_route_success(client, mocker):
    """Test the /send route with a valid movie name."""
    mock_recommendations = ["Movie 1", "Movie 2", "Movie 3"]
    mocker.patch('app.get_recommendations', return_value=mock_recommendations)
    mocker.patch('app.get_data2', return_value=[{"title": "Movie 1"}, {"results": []}])
    response = client.get('/send/Movie 1')
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 3
    assert all("title" in movie for movie in data)

def test_send_route_failure(client, mocker):
    """Test the /send route with an invalid movie name."""
    mocker.patch('app.get_recommendations', return_value=None)
    response = client.get('/send/Invalid Movie')
    assert response.status_code == 200
    data = response.get_json()
    assert data == {"message": "movie not found in database"}

def test_get_movie_by_id_success(client, mocker):
    """Test the /api/movie/<id> route with a valid movie ID."""
    mock_data = {"title": "Movie ID 123", "id": 123}
    mocker.patch('requests.get', return_value=mocker.Mock(json=lambda: mock_data))
    response = client.get('/api/movie/123')
    assert response.status_code == 200
    data = response.get_json()
    assert data["title"] == "Movie ID 123"

def test_get_movie_by_id_failure(client, mocker):
    """Test the /api/movie/<id> route with an invalid movie ID."""
    mocker.patch('requests.get', side_effect=Exception("API error"))
    response = client.get('/api/movie/999')
    assert response.status_code == 404
    data = response.get_json()
    assert data == {"message": "Movie not found"}

def test_user_like(client, auth_token, mysql_db, mocker):
    """Test the /api/user/like route."""
    user_id = save_user('test_user_123', 'test@example.com')
    add_like_movie(user_id, '12345', 'Test Movie', 'http://example.com/poster.jpg')
    mocker.patch('mysql_integration.get_like_movies', return_value=[{'movie_id': '12345', 'movie_title': 'Test Movie'}])
    response = client.get('/api/user/like', headers={'Authorization': f'Bearer {auth_token}'})
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 1
    assert data[0]['movie_title'] == 'Test Movie'

def test_add_user_like(client, auth_token, mysql_db):
    """Test the /api/user/like/add route."""
    user_id = save_user('test_user_123', 'test@example.com')
    data = {
        'movie_id': '12345',
        'movie_title': 'Test Movie',
        'poster_path': 'http://example.com/poster.jpg'
    }
    response = client.post('/api/user/like/add', json=data, headers={'Authorization': f'Bearer {auth_token}'})
    assert response.status_code == 200
    assert response.json['message'] == 'Movie added to watched list successfully'

def test_user_preferences(client, auth_token, mysql_db):
    """Test the /api/user/preferences route."""
    user_id = save_user('test_user_123', 'test@example.com')
    save_user_preferences(user_id, genres=['Action'], languages=['en'])
    response = client.get('/api/user/preferences', headers={'Authorization': f'Bearer {auth_token}'})
    assert response.status_code == 200
    data = response.get_json()
    assert 'genres' in data
    assert 'languages' in data
    assert data['genres'] == ['Action']
    assert data['languages'] == ['en']

def test_save_user_preferences(client, auth_token, mysql_db):
    """Test the /api/user/preferences/save route."""
    user_id = save_user('test_user_123', 'test@example.com')
    data = {'genres': ['Action', 'Drama'], 'languages': ['en', 'fr']}
    response = client.post('/api/user/preferences/save', json=data, headers={'Authorization': f'Bearer {auth_token}'})
    assert response.status_code == 200
    assert response.json['message'] == 'Preferences saved successfully'
    prefs = get_user_preferences(user_id)
    assert set(prefs['genres']) == {'Action', 'Drama'}
    assert set(prefs['languages']) == {'en', 'fr'}

def test_recommendations_based_on_watched(client, auth_token, mysql_db, mocker):
    """Test the /api/user/recommendations/based-on-watched route."""
    user_id = save_user('test_user_123', 'test@example.com')
    add_like_movie(user_id, '12345', 'Test Movie', 'http://example.com/poster.jpg')
    mock_recommendations = ['Movie 1', 'Movie 2', 'Movie 3']
    mocker.patch('app.get_recommendations', return_value=mock_recommendations)
    mocker.patch('app.get_data2', return_value=[{"title": "Movie 1"}, {"results": []}])
    response = client.get('/api/user/recommendations/based-on-watched', headers={'Authorization': f'Bearer {auth_token}'})
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 3
    assert all("title" in movie for movie in data)

def test_user_watchlist_sync(client, auth_token, mysql_db):
    """Test the /api/user/watchlist/sync route."""
    user_id = save_user('test_user_123', 'test@example.com')
    data = {'watchlist': [{'movie_id': '12345', 'movie_title': 'Test Movie', 'poster_path': 'http://example.com/poster.jpg'}]}
    response = client.post('/api/user/watchlist/sync', json=data, headers={'Authorization': f'Bearer {auth_token}'})
    assert response.status_code == 200
    assert response.json['message'] == 'Watchlist synced successfully'
    watchlist = get_watchlist(user_id)
    assert len(watchlist) == 1
    assert watchlist[0]['movie_title'] == 'Test Movie'

def test_get_genres(client, mocker):
    """Test the /api/genres route."""
    mock_genres = [{"id": 1, "name": "Action"}, {"id": 2, "name": "Comedy"}]
    mocker.patch('requests.get', return_value=mocker.Mock(json=lambda: {"genres": mock_genres}))
    response = client.get('/api/genres')
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 2
    assert all("name" in genre for genre in data)

def test_get_movie_showtimes(client, mocker):
    """Test the /api/movies/<id>/showtimes route with a valid date."""
    mock_showtimes = [{"id": 1, "movie_id": "12345", "show_date": "2025-05-29", "show_time": "14:00:00", "theater_name": "Theater 1"}]
    mocker.patch('app.get_showtimes', return_value=mock_showtimes)
    response = client.get('/api/movies/12345/showtimes?date=2025-05-29')
    assert response.status_code == 200
    data = response.get_json()
    assert "showtimes" in data
    assert len(data["showtimes"]) == 1

def test_get_seat_availability(client, mocker):
    """Test the /api/showtimes/<id>/seats route."""
    mocker.patch('app.get_booked_seats', return_value=['A1', 'A2'])
    response = client.get('/api/showtimes/1/seats')
    assert response.status_code == 200
    data = response.get_json()
    assert "booked_seats" in data
    assert data["booked_seats"] == ['A1', 'A2']

def test_remove_from_watchlist(client, auth_token, mysql_db):
    """Test the /api/user/watchlist/remove/<int:movie_id> route."""
    user_id = save_user('test_user_123', 'test@example.com')
    add_to_watchlist(user_id, '12345', 'Test Movie', 'http://example.com/poster.jpg')
    response = client.delete('/api/user/watchlist/remove/12345', headers={'Authorization': f'Bearer {auth_token}'})
    assert response.status_code == 200
    assert response.json['message'] == 'Movie removed from watchlist'
    watchlist = get_watchlist(user_id)
    assert len(watchlist) == 0

def test_remove_from_like(client, auth_token, mysql_db):
    """Test the /api/user/like/remove/<int:movie_id> route."""
    user_id = save_user('test_user_123', 'test@example.com')
    add_like_movie(user_id, '12345', 'Test Movie', 'http://example.com/poster.jpg')
    response = client.delete('/api/user/like/remove/12345', headers={'Authorization': f'Bearer {auth_token}'})
    assert response.status_code == 200
    assert response.json['message'] == 'Movie removed from watchlist'  # Lỗi trong app.py, nên sửa thành 'likes'
    likes = get_like_movies(user_id)
    assert len(likes) == 0

def test_auth_verify_success(client, auth_token, mysql_db):
    """Test the /api/auth/verify route with a valid token."""
    user_id = save_user('test_user_123', 'test@example.com', 'Test User')
    response = client.get('/api/auth/verify', headers={'Authorization': f'Bearer {auth_token}'})
    assert response.status_code == 200
    data = response.get_json()
    assert data['valid'] == True
    assert data['user_id'] == user_id
    assert data['email'] == 'test@example.com'
    assert data['name'] == 'Test User'

def test_auth_verify_invalid_token(client, auth_token):
    """Test the /api/auth/verify route with an invalid token."""
    invalid_token = jwt.encode({
        'firebase_uid': 'test_user_123',
        'email': 'test@example.com',
        'exp': datetime.now(timezone.utc) - timedelta(days=1)
    }, app.config['SECRET_KEY'], algorithm="HS256")
    response = client.get('/api/auth/verify', headers={'Authorization': f'Bearer {invalid_token}'})
    assert response.status_code == 401
    assert 'Token is invalid' in response.json['message']

# --- Test for if __name__ == '__main__' (Minimal) ---
def test_app_run_config():
    """Test minimal configuration in if __name__ == '__main__' block."""
    import app
    assert hasattr(app, 'app')
    assert app.app.config['SECRET_KEY'] == 'test-secret-key'  # Kiểm tra trong fixture client


def test_index_route(client):
    """Test the index route returns a template."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"<title>Movie Api</title>" in response.data  # Kiểm tra title trong HTML

def test_get_movie_route_success(client, mocker):
    """Test the /getmovie route with a valid movie name."""
    mock_data = {"title": "Test Movie", "id": 123}
    mocker.patch('app.get_data2', return_value=[mock_data, {"results": [{"key": "trailer_key"}]}])
    response = client.get('/getmovie/Test Movie')
    assert response.status_code == 200
    data = response.get_json()
    assert data[0]["title"] == "Test Movie"

def test_user_profile_success(client, auth_token, mysql_db):
    """Test the /api/user/profile route with a valid token."""
    user_id = save_user('test_user_123', 'test@example.com', 'Test User', 'http://example.com/photo.jpg')
    response = client.get('/api/user/profile', headers={'Authorization': f'Bearer {auth_token}'})
    assert response.status_code == 200
    data = response.get_json()
    assert data['id'] == user_id
    assert data['email'] == 'test@example.com'
    assert data['display_name'] == 'Test User'

def test_user_profile_no_user(client, auth_token, mocker):
    """Test the /api/user/profile route when user is not found."""
    mocker.patch('mysql_integration.get_user_by_firebase_uid', return_value=None)
    response = client.get('/api/user/profile', headers={'Authorization': f'Bearer {auth_token}'})
    assert response.status_code == 401
    data = response.get_json()
    assert data['message'] == 'User not found!'

def test_get_recommendations_existing_movie(client, mocker):
    """Test the /send route with an existing movie."""
    mock_df = pd.DataFrame({
        'title_x': ['Test Movie'],
        'cast': ['Actor 1, Actor 2'],
        'director': ['Director 1'],
        'genres': ['Action'],
        'keywords': ['keyword1'],
        'overview': ['overview']
    })
    mocker.patch('pandas.read_csv', return_value=mock_df)
    mocker.patch('app.get_data', return_value=[
        {'genres': [{'name': 'Action'}], 'overview': 'overview'},
        {'cast': [{'name': 'Actor 1'}], 'crew': [{'job': 'Director', 'name': 'Director 1'}]},
        {'keywords': [{'name': 'keyword1'}]}
    ])
    mocker.patch('app.get_data2', return_value=[{'title': 'Recommended Movie'}, {'results': []}])
    response = client.get('/send/Test Movie')
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 1
    assert data[0]['title'] == 'Recommended Movie'


def test_movie_details_success(client, mocker):
    """Test the /api/movie/<int:movie_id> route with a valid movie ID."""
    mock_data = {"id": 123, "title": "Test Movie", "poster_path": "/poster.jpg"}
    mocker.patch('requests.get', return_value=mocker.Mock(json=lambda: mock_data))
    response = client.get('/api/movie/123')
    assert response.status_code == 200
    data = response.get_json()
    assert data['title'] == 'Test Movie'

def test_movie_details_api_failure(client, mocker):
    """Test the /api/movie/<int:movie_id> route with an API failure."""
    mocker.patch('requests.get', side_effect=requests.exceptions.RequestException("API error"))
    response = client.get('/api/movie/123')
    assert response.status_code == 404
    data = response.get_json()
    assert data['message'] == 'Movie not found'

def test_get_genres_success(client, mocker):
    """Test the /api/genres route with a successful API response."""
    mock_genres = [{"id": 1, "name": "Action"}, {"id": 2, "name": "Comedy"}]
    mocker.patch('requests.get', return_value=mocker.Mock(json=lambda: {"genres": mock_genres}))
    response = client.get('/api/genres')
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 2
    assert data[0]['name'] == 'Action'


def test_movie_showtimes_success(client, mocker):
    """Test the /api/movies/<int:movie_id>/showtimes route with a valid date."""
    mock_showtimes = [{"id": 1, "movie_id": "123", "show_date": "2025-05-29", "show_time": "14:00:00", "theater_name": "Theater 1"}]
    mocker.patch('app.get_showtimes', return_value=mock_showtimes)
    response = client.get('/api/movies/123/showtimes?date=2025-05-29')
    assert response.status_code == 200
    data = response.get_json()
    assert len(data['showtimes']) == 1
    assert data['showtimes'][0]['theater_name'] == 'Theater 1'

def test_movie_showtimes_missing_date(client, mocker):
    """Test the /api/movies/<int:movie_id>/showtimes route with missing date."""
    response = client.get('/api/movies/123/showtimes')
    assert response.status_code == 400
    data = response.get_json()
    assert data['message'] == 'Date is required'

def test_showtime_seats_success(client, mocker):
    """Test the /api/showtimes/<int:showtime_id>/seats route with available seats."""
    mocker.patch('app.get_booked_seats', return_value=['A1', 'A2'])
    response = client.get('/api/showtimes/1/seats')
    assert response.status_code == 200
    data = response.get_json()
    assert data['booked_seats'] == ['A1', 'A2']


def test_recommendations_based_on_watched_no_movies(client, auth_token, mysql_db):
    """Test the /api/user/recommendations/based-on-watched route with no watched movies."""
    user_id = save_user('test_user_123', 'test@example.com')
    response = client.get('/api/user/recommendations/based-on-watched', headers={'Authorization': f'Bearer {auth_token}'})
    assert response.status_code == 200
    data = response.get_json()
    assert data == {"message": "No watched movies found for recommendations"}

def test_register_db_connection_failure(client, mocker):
    """Test the /api/auth/register route with a database connection failure."""
    mocker.patch('mysql_integration.create_connection', return_value=None)
    data = {
        'firebase_uid': 'test_user_123',
        'email': 'test@example.com',
        'name': 'Test User',
        'photo_url': 'http://example.com/photo.jpg'
    }
    response = client.post('/api/auth/register', json=data)
    assert response.status_code == 500
    assert response.json['message'] == 'Failed to save user!'


def test_get_recommendations_existing_movie(client, mocker):
    """Test the /send route with an existing movie."""
    mock_df = pd.DataFrame({
        'title_x': ['Test Movie', 'Recommended Movie'],
        'cast': ['Actor 1, Actor 2', 'Actor 1'],
        'director': ['Director 1', 'Director 1'],
        'genres': ['Action', 'Action'],
        'keywords': ['keyword1', 'keyword1'],
        'overview': ['overview', 'overview']
    })
    mocker.patch('app.pd.read_csv', return_value=mock_df)
    mocker.patch('app.get_data', return_value=[
        {'genres': [{'name': 'Action'}], 'overview': 'overview'},
        {'cast': [{'name': 'Actor 1'}], 'crew': [{'job': 'Director', 'name': 'Director 1'}]},
        {'keywords': [{'name': 'keyword1'}]}
    ])
    mocker.patch('app.cosine_similarity_manual', return_value=np.array([[1.0, 0.9], [0.9, 1.0]]))
    mocker.patch('app.get_data2', return_value=[{'title': 'Recommended Movie'}, {'results': []}])
    response = client.get('/send/Test Movie')
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 1
    assert data[0]['title'] == 'Recommended Movie'

def test_recommendations_based_on_watched_no_movies(client, auth_token):
    user_id = save_user('test_user_123', 'test@example.com')
    response = client.get('/api/user/recommendations/based-on-watched', headers={'Authorization': f'Bearer {auth_token}'})
    assert response.status_code == 200
    data = response.get_json()
    assert data == [] 

