import json

def check_credentials_json(path="credentials.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Basic validity check
        required_keys = {"client_id", "client_secret", "redirect_uris"}
        actual_keys = set(data.get("installed", {}).keys())

        if not required_keys.issubset(actual_keys):
            raise ValueError("Missing required OAuth keys.")

        print("credentials.json is valid.")
        print(f"Client ID: {data['installed']['client_id']}")
    except Exception as e:
        print("credentials.json is invalid.")
        print(f"Reason: {e}")

check_credentials_json()
