# Ishihara screening assets
No validated, licensed plate images were provided with the image recoloring dataset.
The application deliberately reports screening unavailable until these are installed.
Do not generate arbitrary dot plates and label them clinically validated.

Create `plates.json` with `{"plates": [{"id": "plate-01", "image": "data:image/png;base64,...", "options": ["12", "Nothing"], "normal_answer": "12"}]}`.
Use the test publisher's licensed raster plates and administration protocol.
The backend owns answer keys; clients submit only plate IDs and answers.
Current scoring reports errors only, never a diagnosis, subtype or severity.
Ishihara cannot determine tritan deficiency. Clinical subtype classification requires
an appropriate validated protocol and professional assessment beyond this app.
