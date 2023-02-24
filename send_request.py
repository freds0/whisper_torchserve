import argparse
import requests
import time

IP_SERVER = "localhost"
PORT_SERVER = "8888"
 
def send_request(audio_filepath):

    with open(audio_filepath, "rb") as f:
        audio_bytes = f.read()

    st = time.perf_counter()
    res = requests.post(
        "http://{}:{}/predictions/whisper_base".format(IP_SERVER, PORT_SERVER), 
        files={"data": audio_bytes})
    print(res)
    assert res.status_code == 200
    print(f"Time: {time.perf_counter() - st}")
    transcription =  res.text.lower()
    return transcription

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=None, required=True, help='Audio filepath')
    args = parser.parse_args()

    if args.input is None:
        print("Input filepath is required!")
        exit()

    transcription = send_request(audio_filepath=args.input)
    print(transcription)


if __name__ == "__main__":
    main()

