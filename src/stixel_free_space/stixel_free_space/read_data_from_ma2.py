from pathlib import Path
import xml.etree.ElementTree as ET

#files/2023-07-11_12-20-43_28170706_HD1080_FPS15.svo scenario Target undocking (MA2 driving faster, MA2 starting further away)
ROOT = (
    Path.home()
    / "Documents/Data_For_Stixel_based_Free_space_estimation"
    / "2023-07-11_Multi_ZED_Summer"
)

FILES = [
    ROOT / "Spatial Dual/SpatialDualLog_23-07-11_10-21-15/GoogleEarthTrack.kml", 
    ROOT / "KBox2/LOG00054.kml",
    ROOT / "KBox1/LOG00030.kml",
    ROOT / "NTBox/gnssdump-20230710173603.kml",
]

for file in FILES:
    print(f"\n{'=' * 65}")
    print(file.name, "|", file.parent.name)

    if not file.exists():
        print("File missing")
        continue

    tree = ET.parse(file)
    root = tree.getroot()

    coords = []

    for element in root.iter():
        tag = element.tag.split("}")[-1]

        if tag != "coordinates" or not element.text:
            continue

        raw = element.text.strip()

        if "," in raw:
            # Standard KML: longitude,latitude,altitude
            for entry in raw.split():
                values = entry.split(",")
                if len(values) >= 2:
                    lon, lat = map(float, values[:2])
                    alt = float(values[2]) if len(values) > 2 else None
                    coords.append((lat, lon, alt))

        else:
            # Spatial Dual format: longitude latitude
            values = raw.split()
            if len(values) >= 2:
                lon, lat = map(float, values[:2])
                alt = float(values[2]) if len(values) > 2 else None
                coords.append((lat, lon, alt))

    print("Coordinate count:", len(coords))
    print("First five coordinates:")
    for lat, lon, alt in coords[:5]:
        print(f"  Latitude: {lat:.9f}, Longitude: {lon:.9f}, Altitude: {alt}")

    for element in root.iter():
        if element.tag.split("}")[-1] == "when" and element.text:
            print("First timestamp:", element.text)
            break