"""
Direct API tests using Flask test client for Step 2 of build order.
Tests:
- POST /api/check-risk with Hyderabad
- POST /api/check-risk with Delhi
- GET /api/forecast?location=Hyderabad
- GET /api/forecast?location=Chennai
"""

import json
from app import app

client = app.test_client()

print("--- Testing POST /api/check-risk with Hyderabad ---")
res1 = client.post("/api/check-risk", json={
    "location": "Hyderabad",
    "age": 34,
    "acclimatization": "not",
    "workload": "heavy",
    "region": "plains"
})
print("Status:", res1.status_code)
data1 = res1.get_json()
print("Keys:", list(data1.keys()))
print("Result:", {
    "location": data1.get("location"),
    "temp": data1.get("current_temp_c"),
    "wbgt": data1.get("wbgt_c"),
    "wbgt_band": data1.get("wbgt_band"),
    "imd": data1.get("imd_status"),
    "risk": data1.get("risk_category"),
    "work": data1.get("work_minutes"),
    "rest": data1.get("rest_minutes"),
})
assert res1.status_code == 200
assert "risk_category" in data1
assert "wbgt_c" in data1
assert "work_minutes" in data1

print("\n--- Testing POST /api/check-risk with Delhi ---")
res2 = client.post("/api/check-risk", json={
    "location": "Delhi",
    "age": 70,
    "acclimatization": "partially",
    "workload": "light",
    "region": "plains"
})
print("Status:", res2.status_code)
data2 = res2.get_json()
print("Result:", {
    "location": data2.get("location"),
    "temp": data2.get("current_temp_c"),
    "wbgt": data2.get("wbgt_c"),
    "wbgt_band": data2.get("wbgt_band"),
    "imd": data2.get("imd_status"),
    "risk": data2.get("risk_category"),
    "work": data2.get("work_minutes"),
    "rest": data2.get("rest_minutes"),
})
assert res2.status_code == 200

print("\n--- Testing GET /api/forecast?location=Hyderabad ---")
res3 = client.get("/api/forecast?location=Hyderabad")
print("Status:", res3.status_code)
data3 = res3.get_json()
print("Count of days:", len(data3))
assert len(data3) == 5
print("Day 1 sample:", {
    "date": data3[0].get("date"),
    "temp": data3[0].get("current_temp_c"),
    "wbgt": data3[0].get("wbgt_c"),
    "risk": data3[0].get("risk_category"),
    "severity": data3[0].get("severity"),
})

print("\n--- Testing GET /api/forecast?location=Chennai ---")
res4 = client.get("/api/forecast?location=Chennai")
print("Status:", res4.status_code)
data4 = res4.get_json()
print("Count of days:", len(data4))
assert len(data4) == 5
print("Day 1 sample:", {
    "date": data4[0].get("date"),
    "temp": data4[0].get("current_temp_c"),
    "wbgt": data4[0].get("wbgt_c"),
    "risk": data4[0].get("risk_category"),
    "severity": data4[0].get("severity"),
})

print("\nALL API TESTS PASSED SUCCESSFULLY!")
