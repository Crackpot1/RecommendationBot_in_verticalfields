import streamlit as st
import json
from typing import Optional
from streamlit.components.v1 import html
import uuid
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import io
import speech_recognition as sr

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import io
import speech_recognition as sr
from queue import Queue

# 全局线程安全队列（替代直接使用 session_state）
_audio_queue = Queue()


def _callback(indata, frames, time, status):
    """声音采集回调函数（不直接使用 session_state）"""
    _audio_queue.put(indata.copy())


def start_voice_input(language):
    """开始录音并返回流对象"""
    # 初始化音频队列
    global _audio_queue
    _audio_queue = Queue()

    # 创建音频流
    stream = sd.InputStream(
        samplerate=16000,
        channels=1,
        dtype='float32',
        callback=_callback
    )
    stream.start()
    return stream


def stop_voice_input(stream, language="en-US"):
    """停止录音并返回识别文本"""
    stream.stop()
    stream.close()

    # 从队列中获取所有音频数据
    audio_data = []
    while not _audio_queue.empty():
        audio_data.append(_audio_queue.get())

    if len(audio_data) == 0:
        return ""

    # 合并音频数据
    audio_array = np.concatenate(audio_data, axis=0)

    # 转换为WAV格式
    wav_io = io.BytesIO()
    write(wav_io, 16000, (audio_array * 32767).astype(np.int16))
    wav_io.seek(0)

    # 语音识别
    r = sr.Recognizer()
    try:
        with sr.AudioFile(wav_io) as source:
            audio = r.record(source)
        return r.recognize_google(audio, language=language.split("-")[0])
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "API unavailable"


# 语音输出组件
def text_to_speech(text: str, language: str = "en-US"):
    """使用浏览器SpeechSynthesis实现文本朗读"""

    # 生成安全JSON字符串
    safe_text = json.dumps(text).replace('"', "'")

    # 语音合成JS代码
    js_code = f"""
    <script>
    function speak() {{
        if ('speechSynthesis' in window) {{
            const utterance = new SpeechSynthesisUtterance();
            utterance.text = {safe_text};
            utterance.lang = "{language}";
            utterance.rate = 1.0;
            window.speechSynthesis.speak(utterance);
        }} else {{
            console.error('Speech synthesis not supported');
        }}
    }}

    // 自动触发语音播放
    window.addEventListener('load', speak);
    </script>
    """

    # 注入JavaScript代码
    st.components.v1.html(
        js_code,
        height=0,
        width=0,
    )
