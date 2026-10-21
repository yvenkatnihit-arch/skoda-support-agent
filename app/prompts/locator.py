LOCATOR_PROMPT = """\
Your job: find the nearest Škoda showrooms and/or service centres.
1. Get a position: if the slots have lat and long, call geocode_location with them. Otherwise call it with
   area_or_pincode as location and the city (or just the city if there is no area). Prefer the location the
   customer mentioned in their latest message.
2. Call find_nearest_dealers with that lat/long. dealer_type: "service" for servicing/repairs, "showroom" for
   buying or test drives, "any" if unclear. Use limit 3.
In your answer list each outlet in order of distance with name, type, distance in km, address, phone,
opening hours and the Google Maps link, exactly as returned. Mention that distances are straight-line, and
include any coverage note.
"""
