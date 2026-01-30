#!/usr/bin/env python3
"""Test script for PRIMORDIAL web interface"""

import json
import time
import requests
import websocket
import threading

BASE_URL = "http://127.0.0.1:8080"
WS_URL = "ws://127.0.0.1:8080/ws"

def test_api():
    """Test REST API endpoints"""
    print("\n=== Testing REST API ===")

    # Test state
    try:
        r = requests.get(f"{BASE_URL}/api/state")
        print(f"GET /api/state: {r.status_code} - {r.json()}")
    except Exception as e:
        print(f"GET /api/state: ERROR - {e}")

    # Test settings
    try:
        r = requests.get(f"{BASE_URL}/api/settings")
        settings = r.json()
        print(f"GET /api/settings: {r.status_code} - grid_size={settings.get('grid_size')}, pop={settings.get('initial_population')}")
    except Exception as e:
        print(f"GET /api/settings: ERROR - {e}")

    return settings

def test_update_settings(grid_size=255):
    """Test updating settings"""
    print(f"\n=== Testing Update Settings (grid_size={grid_size}) ===")

    # Get current settings
    r = requests.get(f"{BASE_URL}/api/settings")
    settings = r.json()

    # Modify grid size
    settings['grid_size'] = grid_size
    # Scale population
    scale = (grid_size * grid_size) / (80 * 80)
    settings['initial_population'] = int(500 * scale)
    settings['max_population'] = int(900 * scale)

    print(f"Sending: grid_size={settings['grid_size']}, pop={settings['initial_population']}")

    try:
        r = requests.post(f"{BASE_URL}/api/settings", json=settings)
        print(f"POST /api/settings: {r.status_code}")
    except Exception as e:
        print(f"POST /api/settings: ERROR - {e}")

def test_websocket(duration=5):
    """Test WebSocket connection and receive snapshots"""
    print(f"\n=== Testing WebSocket (listening for {duration}s) ===")

    messages = []

    def on_message(ws, message):
        data = json.loads(message)
        msg_type = data.get('type')
        if msg_type == 'Snapshot':
            print(f"  Snapshot: time={data.get('time')}, grid={data.get('grid_size')}, orgs={len(data.get('organisms', []))}")
        elif msg_type == 'StateChange':
            print(f"  StateChange: {data.get('state')}")
        messages.append(data)

    def on_error(ws, error):
        print(f"  WebSocket error: {error}")

    def on_close(ws, close_status, close_msg):
        print(f"  WebSocket closed")

    def on_open(ws):
        print(f"  WebSocket connected!")

    ws = websocket.WebSocketApp(WS_URL,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open)

    # Run in thread with timeout
    thread = threading.Thread(target=ws.run_forever)
    thread.daemon = True
    thread.start()

    time.sleep(duration)
    ws.close()

    print(f"  Received {len(messages)} messages")
    return messages

def test_resume():
    """Test resume command"""
    print("\n=== Testing Resume ===")
    try:
        r = requests.post(f"{BASE_URL}/api/sim/resume")
        print(f"POST /api/sim/resume: {r.status_code}")
    except Exception as e:
        print(f"POST /api/sim/resume: ERROR - {e}")

if __name__ == "__main__":
    import sys

    print("PRIMORDIAL Web Interface Test")
    print("=" * 40)

    # Test API
    settings = test_api()

    # Test WebSocket
    print("\n--- Initial state ---")
    test_websocket(3)

    # If argument provided, test that grid size
    if len(sys.argv) > 1:
        grid_size = int(sys.argv[1])
        test_update_settings(grid_size)
        time.sleep(1)
        test_resume()
        print("\n--- After update ---")
        test_websocket(5)

    print("\n=== Test Complete ===")
