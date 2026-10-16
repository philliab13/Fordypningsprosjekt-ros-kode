from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter

file = (
    Path.home()
    / "Documents/Data_For_Stixel_based_Free_space_estimation"
    / "2023-07-11_Multi_ZED_Summer"
    / "Spatial Dual/SpatialDualLog_23-07-11_10-21-15"
    / "GoogleEarthTrack.kml"
)

tree = ET.parse(file)
root = tree.getroot()

print("File size:", file.stat().st_size, "bytes")

print("\nXML element types:")
tags = Counter(
    element.tag.split("}")[-1]
    for element in root.iter()
)
for tag, count in tags.most_common():
    print(f"{tag}: {count}")

print("\nFirst 5000 characters:")
print(file.read_text(errors="replace")[:5000])