import os, requests
from dotenv import load_dotenv

def main():
    load_dotenv()
    print("Hello from Codex environment!")
    # Example: call a public API (only works if you enable internet access)
    # r = requests.get("https://api.github.com/rate_limit")
    # print(r.json())

if __name__ == "__main__":
    main()
