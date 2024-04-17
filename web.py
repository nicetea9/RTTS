import subprocess
import time
import streamlit as st
import streamlink
from langchain.chat_models.gigachat import GigaChat
from speechkit import configure_credentials, creds, model_repository
from speechkit.stt import AudioProcessingType
from langchain.schema import HumanMessage, SystemMessage
from clearml import Task


# Определение функции process_stream
def process_stream(url):
    try:
        task = Task.init(project_name='RTTS', task_name='experiment')
        all_result = []
        streams = streamlink.streams(url)
        audio_stream = streams["audio_only"]
        configure_credentials(
            yandex_credentials=creds.YandexCredentials(api_key='AQVN3qwuRDroqa-9P2bAlSWuh_UVmVe_CiOHmEdF')
        )
        model = model_repository.recognition_model()
        model.model = 'general:rc'
        model.language = 'ru-RU'
        n = "c"
        model.audio_processing_type = AudioProcessingType.Full
        for i in range(3):
            duration = 30
            start = i * duration
            output = f"fragmen({n})_{i + 1}.mp3"
            ffmpeg = [
                "ffmpeg",
                "-i", audio_stream.url,
                "-ss", str(start),
                "-t", str(duration),
                "-vn",
                "-c", "copy",
                "-acodec", "libmp3lame",
                output
            ]
            subprocess.run(ffmpeg)
            time.sleep(2)
            audio_artifact = task.upload_artifact(name='Custom_audio', artifact_object=output)
            task.get_logger().report_media(
                title='audio',
                series='tada',
                iteration=1,
                local_path=output
            )
            result = model.transcribe_file(output)
            all_result.append(result[0].normalized_text)  # Добавляем результат сокращения в список
            print(f"Text {output}: {result[0].normalized_text}")

        all_result_str = '\n'.join(all_result)  # Объединяем результаты сокращения в одну строку

        client_id = "1a8c7831-f59c-4904-9455-608bc1a3f2af"
        client_secret = "0a6fc530-79bd-4497-b37f-bf2cc10e5c97"
        authorization_data = "MWE4Yzc4MzEtZjU5Yy00OTA0LTk0NTUtNjA4YmMxYTNmMmFmOjBhNmZjNTMwLTc5YmQtNDQ5Ny1iMzdmLWJmMmNjMTBlNWM5Nw=="

        chat = GigaChat(credentials=authorization_data, verify_ssl_certs=False, scope="GIGACHAT_API_PERS")

        messages = [
            SystemMessage(
                content="Сократи текст:"
            )
        ]
        messages.append(HumanMessage(content=all_result_str))
        res = chat(messages)
        messages.append(res)
        short = open("short.txt", "w+")
        short.write(res.content)
        print(res.content)

        text_artifact = task.upload_artifact(name='custom_text', artifact_object='short.txt')
        task.get_logger().report_media(
            title='text',
            series='documents',
            iteration=1,
            local_path='short.txt'
        )
        task.close()

        return all_result_str  # Возвращаем сокращенный текст
    except Exception as e:
        st.error(f'Произошла ошибка: {e}')

    # Заголовок приложения


st.markdown("<h1 style='text-align: center; color: blue;'>RTTS</h1>", unsafe_allow_html=True)

# Получение URL от пользователя
url = st.sidebar.text_input('Введите URL трансляции', '')

# Обработка нажатия кнопки "Отправить"
if st.sidebar.button('Отправить'):
    # Здесь вызывайте ваш алгоритм для обработки URL и аудиопотока
    # Например:
    processed_text = process_stream(url)
    # Отображение сокращенного текста под надписью RTTS в прямоугольном окне
    st.markdown("<div class='rectangle'>Сокращенный текст:<br>{}</div>".format(processed_text), unsafe_allow_html=True)

# Дополнительные стили для страницы
st.write(""" 
    <style> 
    .rectangle { 
        width: 700px; 
        height: 550px; 
        border: 1px solid #ccc; 
        background-color: transparent; 
        margin: auto; 
        padding: 20px; 
    } 
    </style>
    """, unsafe_allow_html=True)
