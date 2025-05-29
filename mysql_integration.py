import pandas as pd
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import mysql.connector
from mysql.connector import Error
import hashlib
import uuid
import json
from datetime import datetime, timedelta

# Kết nối MySQL
def create_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='movie_film',
            user='root',
            password=''
        )
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return None

# Khởi tạo database nếu chưa có
def initialize_database():
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            
            # Tạo bảng users nếu chưa tồn tại
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    firebase_uid VARCHAR(255) UNIQUE NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    display_name VARCHAR(255),
                    photo_url VARCHAR(1024),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB
            ''')
            
            # Tạo bảng user_preferences với cấu trúc mới
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    preference_type VARCHAR(50) NOT NULL,
                    preference_value VARCHAR(255) NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                ) ENGINE=InnoDB
            ''')
            
            # Tạo bảng watchlist với cột poster_path
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS watchlist (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    movie_id VARCHAR(255) NOT NULL,
                    movie_title VARCHAR(255) NOT NULL,
                    poster_path TEXT,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    UNIQUE (user_id, movie_id)
                ) ENGINE=InnoDB
            ''')
            
            # Tạo bảng watched với cột poster_path
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS likes (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    movie_id VARCHAR(255) NOT NULL,
                    movie_title VARCHAR(255) NOT NULL,
                    poster_path VARCHAR(1024),
                    watched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    UNIQUE (user_id, movie_id)
                ) ENGINE=InnoDB
            ''')
            
            # Tạo bảng showtimes để lưu thông tin lịch chiếu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS showtimes (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    movie_id VARCHAR(255) NOT NULL,
                    show_date DATE NOT NULL,
                    show_time TIME NOT NULL,
                    theater_name VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB
            ''')
            
            # Tạo bảng bookings để lưu thông tin đặt vé
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS bookings (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    showtime_id INT NOT NULL,
                    seat_ids LONGTEXT NOT NULL,
                    total_price DECIMAL(10,2) NOT NULL,
                    payment_method VARCHAR(50) NOT NULL,
                    payment_status ENUM('pending', 'completed', 'failed') DEFAULT 'pending',
                    booking_status ENUM('confirmed', 'cancelled') DEFAULT 'confirmed',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (showtime_id) REFERENCES showtimes(id) ON DELETE CASCADE
                ) ENGINE=InnoDB
            ''')
            
            connection.commit()
            print("Database initialized successfully")
        except Error as e:
            print(f"Error initializing database: {e}")
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

# Thêm/cập nhật thông tin người dùng sau khi đăng nhập Firebase
def save_user(firebase_uid, email, display_name=None, photo_url=None):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            
            cursor.execute("SELECT id FROM users WHERE firebase_uid = %s", (firebase_uid,))
            result = cursor.fetchone()
            
            if result:
                user_id = result[0]
                cursor.execute("""
                    UPDATE users 
                    SET email = %s, display_name = %s, photo_url = %s, last_login = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (email, display_name, photo_url, user_id))
            else:
                cursor.execute("""
                    INSERT INTO users (firebase_uid, email, display_name, photo_url)
                    VALUES (%s, %s, %s, %s)
                """, (firebase_uid, email, display_name, photo_url))
                
                cursor.execute("SELECT id FROM users WHERE firebase_uid = %s", (firebase_uid,))
                user_id = cursor.fetchone()[0]
            
            connection.commit()
            return user_id
        except Error as e:
            print(f"Error saving user: {e}")
            connection.rollback()
            return None
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

# Lấy thông tin người dùng bằng Firebase UID
def get_user_by_firebase_uid(firebase_uid):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT id, firebase_uid, email, display_name, photo_url, 
                       created_at, last_login 
                FROM users 
                WHERE firebase_uid = %s
            """, (firebase_uid,))
            user = cursor.fetchone()
            return user
        except Error as e:
            print(f"Error getting user: {e}")
            return None
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

# Thêm phim vào danh sách muốn xem
def add_to_watchlist(user_id, movie_id, movie_title, poster_path=''):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("""
                INSERT INTO watchlist (user_id, movie_id, movie_title, poster_path)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE added_at = CURRENT_TIMESTAMP, poster_path = %s
            """, (user_id, movie_id, movie_title, poster_path, poster_path))
            connection.commit()
            return True
        except Error as e:
            print(f"Error adding to watchlist: {e}")
            connection.rollback()
            return False
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

# Lấy danh sách phim muốn xem của người dùng
def get_watchlist(user_id):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT movie_id, movie_title, poster_path, added_at
                FROM watchlist
                WHERE user_id = %s
                ORDER BY added_at DESC
            """, (user_id,))
            watchlist = cursor.fetchall()
            return watchlist
        except Error as e:
            print(f"Error getting watchlist: {e}")
            return []
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

# Thêm phim đã xem
def add_like_movie(user_id, movie_id, movie_title, poster_path=''):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("""
                INSERT INTO likes (user_id, movie_id, movie_title, poster_path)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE watched_at = CURRENT_TIMESTAMP, poster_path = %s
            """, (user_id, movie_id, movie_title, poster_path, poster_path))
            
            cursor.execute("""
                DELETE FROM watchlist 
                WHERE user_id = %s AND movie_id = %s
            """, (user_id, movie_id))
            
            connection.commit()
            return True
        except Error as e:
            print(f"Error adding watched movie: {e}")
            connection.rollback()
            return False
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

# Lấy danh sách phim đã xem
def get_like_movies(user_id):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT movie_id, movie_title, poster_path, watched_at
                FROM likes
                WHERE user_id = %s
                ORDER BY watched_at DESC
            """, (user_id,))
            watched = cursor.fetchall()
            return watched
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

def save_user_preferences(user_id, genres=None, languages=None):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            
            cursor.execute("DELETE FROM user_preferences WHERE user_id = %s", (user_id,))
            
            if genres:
                for genre in genres:
                    cursor.execute("""
                        INSERT INTO user_preferences (user_id, preference_type, preference_value)
                        VALUES (%s, 'genre', %s)
                    """, (user_id, genre))
            
            if languages:
                for language in languages:
                    cursor.execute("""
                        INSERT INTO user_preferences (user_id, preference_type, preference_value)
                        VALUES (%s, 'language', %s)
                    """, (user_id, language))
                
            connection.commit()
            return True
        except Error as e:
            print(f"Error saving preferences: {e}")
            connection.rollback()
            return False
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

def get_user_preferences(user_id):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            
            cursor.execute("""
                SELECT preference_value 
                FROM user_preferences
                WHERE user_id = %s AND preference_type = 'genre'
            """, (user_id,))
            genres = [row['preference_value'] for row in cursor.fetchall()]
            
            cursor.execute("""
                SELECT preference_value 
                FROM user_preferences
                WHERE user_id = %s AND preference_type = 'language'
            """, (user_id,))
            languages = [row['preference_value'] for row in cursor.fetchall()]
            
            return {
                'genres': genres,
                'languages': languages
            }
        except Error as e:
            print(f"Error getting preferences: {e}")
            return {'genres': [], 'languages': []}
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

# Thêm hàm để lưu thông tin lịch chiếu
def save_showtime(movie_id, show_date, show_time, theater_name):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("""
                INSERT INTO showtimes (movie_id, show_date, show_time, theater_name)
                VALUES (%s, %s, %s, %s)
            """, (movie_id, show_date, show_time, theater_name))
            connection.commit()
            cursor.execute("SELECT LAST_INSERT_ID()")
            showtime_id = cursor.fetchone()[0]
            return showtime_id
        except Error as e:
            print(f"Error saving showtime: {e}")
            connection.rollback()
            return None
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

def timedelta_to_str(td):
    if isinstance(td, timedelta):
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return str(td)

# Thêm hàm để lấy danh sách lịch chiếu
def get_showtimes(movie_id, show_date):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT id, movie_id, show_date, show_time, theater_name
                FROM showtimes
                WHERE movie_id = %s AND show_date = %s
                ORDER BY show_time
            """, (movie_id, show_date))
            showtimes = cursor.fetchall()
            
            for showtime in showtimes:
                if 'show_time' in showtime and showtime['show_time'] is not None:
                    showtime['show_time'] = timedelta_to_str(showtime['show_time'])
            return showtimes
        except Error as e:
            print(f"Error getting showtimes: {e}")
            return []
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

# Thêm hàm để lấy thông tin ghế đã đặt
def get_booked_seats(showtime_id):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("""
                SELECT seat_ids
                FROM bookings
                WHERE showtime_id = %s AND booking_status = 'confirmed'
            """, (showtime_id,))
            booked_seats = []
            for row in cursor.fetchall():
                seat_ids = json.loads(row[0])
                booked_seats.extend(seat_ids)
            return booked_seats
        except Error as e:
            print(f"Error getting booked seats: {e}")
            return []
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

# Thêm hàm để lưu thông tin đặt vé
def save_booking(user_id, showtime_id, seat_ids, total_price, payment_method):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("""
                INSERT INTO bookings (user_id, showtime_id, seat_ids, total_price, payment_method, payment_status)
                VALUES (%s, %s, %s, %s, %s, 'completed')
            """, (user_id, showtime_id, json.dumps(seat_ids), total_price, payment_method))
            connection.commit()
            cursor.execute("SELECT LAST_INSERT_ID()")
            booking_id = cursor.fetchone()[0]
            return booking_id
        except Error as e:
            print(f"Error saving booking: {e}")
            connection.rollback()
            return None
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

# Thêm hàm để lấy danh sách vé đã đặt của người dùng
def get_user_bookings(user_id):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT b.id, b.showtime_id, b.seat_ids, b.total_price, b.payment_method, 
                       b.payment_status, b.booking_status, b.created_at,
                       s.movie_id, s.show_date, s.show_time, s.theater_name
                FROM bookings b
                JOIN showtimes s ON b.showtime_id = s.id
                WHERE b.user_id = %s AND b.booking_status = 'confirmed'
                ORDER BY b.created_at DESC
            """, (user_id,))
            bookings = cursor.fetchall()
            for booking in bookings:
                booking['seat_ids'] = json.loads(booking['seat_ids'])
            return bookings
        except Error as e:
            print(f"Error getting bookings: {e}")
            return []
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

# Khởi tạo database khi khởi động ứng dụng
initialize_database()