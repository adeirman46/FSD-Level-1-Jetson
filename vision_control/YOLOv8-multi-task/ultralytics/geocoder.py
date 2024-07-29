import requests

def get_current_gps_coordinates():
    try:
        response = requests.get('https://ipinfo.io/json')
        data = response.json()
        loc = data.get('loc')
        if loc:
            lat, lon = loc.split(',')
            return float(lat), float(lon)
        else:
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    coordinates = get_current_gps_coordinates()
    if coordinates:
        latitude, longitude = coordinates
        print(f"Your current GPS coordinates are:")
        print(f"Latitude: {latitude}")
        print(f"Longitude: {longitude}")
    else:
        print("Unable to retrieve your GPS coordinates.")

