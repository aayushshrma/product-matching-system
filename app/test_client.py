import requests

def test_match(image_path):
    url = "http://localhost:8000/match"
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)
    print("Response:", response.status_code)
    print(response.json())


if __name__ == "__main__":
    test_match("catalog/boots.jpg")