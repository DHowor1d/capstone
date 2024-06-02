import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="사용할 모델",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--energy_threshold", default=1000,
                        help="마이크가 감지할 에너지 수준.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="녹음이 실시간으로 처리되는 시간(초).", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="녹음 사이의 공백이 얼마나 지속되어야 새로운 줄로 간주할지(초).", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="SpeechRecognition에 사용할 기본 마이크 이름. "
                                 "'list'로 실행하여 사용 가능한 마이크를 확인하세요.", type=str)
    args = parser.parse_args()

    # 큐에서 마지막으로 녹음이 검색된 시간.
    phrase_time = None
    # 스레드 안전 큐로, 스레드 녹음 콜백에서 데이터를 전달합니다.
    data_queue = Queue()
    # 음성이 끝났을 때를 감지할 수 있는 기능이 있는 SpeechRecognizer를 사용하여 오디오를 녹음합니다.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # 동적 에너지 임계값 조정을 비활성화합니다. 이 옵션을 활성화하면 SpeechRecognizer가 녹음을 멈추지 않게 됩니다.
    recorder.dynamic_energy_threshold = False

    # 리눅스 사용자를 위한
    # 잘못된 마이크를 사용하여 애플리케이션이 영구적으로 멈추거나 충돌하는 것을 방지합니다.
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("사용 가능한 마이크 장치는 다음과 같습니다: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"\"{name}\" 이름의 마이크를 찾았습니다.")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    # 모델 로드 / 다운로드
    model = args.model
    
    # 항상 다국어 모델 불러오기
    audio_model = whisper.load_model(model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        녹음이 끝날 때 오디오 데이터를 받는 스레드 콜백 함수
        audio: 녹음된 바이트를 포함하는 AudioData.
        """
        # 원시 바이트를 잡아서 스레드 안전 큐에 넣습니다.
        data = audio.get_raw_data()
        data_queue.put(data)

    # 백그라운드 스레드를 생성합니다.
    # SpeechRecognize 사용
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    print("음성인식을 시작합니다.\n")

    while True:
        try:
            now = datetime.utcnow()
            # 큐에서 녹음된 원시 오디오를 가져옴
            if not data_queue.empty():
                phrase_complete = False
                # 녹음 사이에 충분한 시간이 지났다면, 구문을 완성된 것으로 간주
                # 현재 작업 중인 오디오 버퍼를 지워서 새 데이터로 시작
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                # 큐에서 새 오디오 데이터를 받은 마지막 시간.
                phrase_time = now
                
                # 큐에서 오디오 데이터를 결합
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                
                # 램 버퍼에서 모델이 직접 사용할 수 있는 것으로 변환
                # 데이터를 16비트 정수에서 32비트 부동 소수점으로 변환
                # 오디오 스트림 주파수를 PCM 파장과 호환되는 최대 32768hz로 제한
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # 전사를 읽습니다.
                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                # 녹음 사이에 일시 중지를 감지했다면, 전사에 새 항목을 추가합니다.
                # 그렇지 않다면 기존 것을 편집합니다.
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                # 콘솔을 지우고 업데이트된 전사를 다시 출력합니다.
                os.system('cls' if os.name=='nt' else 'clear')
                for line in transcription:
                    print(line)
                # 출력을 갱신합니다.
                print('', end='', flush=True)
            else:
                # 프로세스 쉬게 하기
                sleep(0.2)
        except KeyboardInterrupt:
            break

    print("\n\n인식 결과:")
    for line in transcription:
        print(line)


if __name__ == "__main__":
    main()
