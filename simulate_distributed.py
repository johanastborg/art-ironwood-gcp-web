import sys
import os
import queue
import threading
import time
import json
from unittest.mock import MagicMock, patch
import logging

# Ensure project root is in path
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simulator")

# --- Mocking Infrastructure ---

class MessageBus:
    def __init__(self):
        self.queues = {}
        # Pre-define queues for known subscriptions
        self.queues["projects/test-project/subscriptions/render-requests-sub"] = queue.Queue()
        self.queues["projects/test-project/subscriptions/render-results-sub"] = queue.Queue()

    def publish(self, topic, data):
        # Route to appropriate queue based on topic
        if "render-requests" in topic:
            q = self.queues["projects/test-project/subscriptions/render-requests-sub"]
            msg = MagicMock()
            msg.data = data
            msg.ack = MagicMock()
            msg.nack = MagicMock()
            q.put(msg)
        elif "render-results" in topic:
            q = self.queues["projects/test-project/subscriptions/render-results-sub"]
            msg = MagicMock()
            msg.data = data
            msg.ack = MagicMock()
            msg.nack = MagicMock()
            q.put(msg)

bus = MessageBus()

class MockPublisherClient:
    def topic_path(self, project, topic):
        return f"projects/{project}/topics/{topic}"

    def publish(self, topic, data, **kwargs):
        bus.publish(topic, data)
        future = MagicMock()
        future.result = MagicMock(return_value="msg_id")
        return future

class MockSubscriberClient:
    def subscription_path(self, project, subscription):
        return f"projects/{project}/subscriptions/{subscription}"

    def subscribe(self, subscription, callback):
        def poll():
            q = bus.queues.get(subscription)
            if not q:
                return
            while True:
                try:
                    msg = q.get(timeout=0.1)
                    callback(msg)
                except queue.Empty:
                    continue
                except Exception as e:
                    break

        t = threading.Thread(target=poll, daemon=True)
        t.start()

        future = MagicMock()
        future.result = MagicMock() # blocks
        future.cancel = MagicMock()
        return future

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

# --- Runner ---

def run_simulation():
    # Patch the google.cloud.pubsub_v1 module where it is used
    # Since we don't know exactly where it's imported from (could be backend.master.main or backend.worker.main)
    # We patch it in `sys.modules` or use `patch` on the library itself if possible.
    # The safest way is to patch before imports.

    with patch("google.cloud.pubsub_v1.PublisherClient", side_effect=MockPublisherClient), \
         patch("google.cloud.pubsub_v1.SubscriberClient", side_effect=MockSubscriberClient):

        print("Starting Simulation...")

        # Import worker and run in thread
        # We need to ensure we don't import them at top level
        from backend.worker.main import main as worker_main

        worker_thread = threading.Thread(target=worker_main, daemon=True)
        worker_thread.start()
        print("Worker started.")

        # Give it a moment to subscribe
        time.sleep(1)

        # Import master app
        from backend.master.main import app
        from fastapi.testclient import TestClient

        with TestClient(app) as client:
            print("Sending request to Master...")
            # Note: backend.master.main expects env var GOOGLE_CLOUD_PROJECT, defaults to test-project

            response = client.post("/render", json={
                "scene": {"objects": []},
                "width": 100,
                "height": 80,
                "samples": 10
            })

        if response.status_code == 200:
            data = response.json()
            print("Render Request Successful.")
            print(f"Job ID: {data.get('job_id')}")
            print(f"Status: {data.get('status')}")
            tiles = data.get('tiles')
            print(f"Tiles received: {len(tiles)}")

            if len(tiles) == 8:
                print("SUCCESS: Received all 8 tiles.")
            else:
                print(f"FAILURE: Expected 8 tiles, got {len(tiles)}")
        else:
            print(f"FAILURE: Request failed with {response.status_code}: {response.text}")

if __name__ == "__main__":
    run_simulation()
