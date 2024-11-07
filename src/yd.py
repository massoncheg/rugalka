# Скачивает видео
import os
import yt_dlp

def get_unique_filename(filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(new_filename):
        new_filename = f"{base}{counter}{ext}"
        counter += 1
    return new_filename

def download_audio(youtube_url, output_filename="audio"):
    # Проверяем и получаем уникальное имя файла
    output_filename = get_unique_filename(output_filename)
    
    ydl_opts = {
        'format': 'bestaudio/best',  # Выбор наилучшего доступного аудиоформата
        'outtmpl': output_filename,  # Название выходного файла
        'noplaylist': True,          # Не скачивать плейлист, только одно видео
        'postprocessors': [{         # Постобработка для конвертации в MP3
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"Скачивание аудио из видео: {youtube_url}")
        ydl.download([youtube_url])
        print(f"Аудиофайл сохранён как {output_filename}")

if __name__ == "__main__":
    youtube_url = input("Введите ссылку на YouTube видео: ")
    output_filename = input("Введите имя для сохранения аудиофайла (например, 'audio.mp3'): ") or "audio.mp3"
    download_audio(youtube_url, output_filename)
