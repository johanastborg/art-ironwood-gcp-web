import os
import uuid
import json
import base64
import asyncio
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import pubsub_v1
from typing import Dict, Any, List, Optional
import logging
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configuration
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "test-project")
TOPIC_REQUESTS = "render-requests"
TOPIC_RESULTS = "render-results"
SUBSCRIPTION_RESULTS = "render-results-sub"

# Global clients (initialized lazily or with error handling)
publisher = None
subscriber = None

def get_publisher():
    global publisher
    if publisher is None:
        try:
            publisher = pubsub_v1.PublisherClient()
        except Exception as e:
            logger.warning(f"Could not init Publisher: {e}")
    return publisher

def get_subscriber():
    global subscriber
    if subscriber is None:
        try:
            subscriber = pubsub_v1.SubscriberClient()
        except Exception as e:
            logger.warning(f"Could not init Subscriber: {e}")
    return subscriber

# In-memory store for pending jobs
jobs: Dict[str, Any] = {}
job_lock = threading.Lock()

class RenderRequest(BaseModel):
    scene: Dict[str, Any]
    width: int = 800
    height: int = 600
    samples: int = 1024

def result_callback(message):
    try:
        data = json.loads(message.data.decode("utf-8"))
        job_id = data.get("job_id")
        tile_index = data.get("tile_index")
        result_b64 = data.get("result")

        message.ack()

        with job_lock:
            if job_id in jobs:
                job = jobs[job_id]
                job["tiles"][tile_index] = result_b64

                if len(job["tiles"]) == job["total_tiles"]:
                    loop = job["loop"]
                    future = job["future"]
                    if not future.done():
                        loop.call_soon_threadsafe(future.set_result, True)

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        message.nack()

@app.on_event("startup")
async def startup_event():
    sub = get_subscriber()
    if sub:
        subscription_path = sub.subscription_path(PROJECT_ID, SUBSCRIPTION_RESULTS)
        try:
            # We don't await this, it runs in background threads
            sub.subscribe(subscription_path, callback=result_callback)
            logger.info(f"Listening on {subscription_path}")
        except Exception as e:
            logger.error(f"Failed to subscribe: {e}")

@app.post("/render")
async def render(request: RenderRequest):
    pub = get_publisher()
    if not pub:
        raise HTTPException(status_code=500, detail="PubSub Publisher not available")

    job_id = str(uuid.uuid4())
    width = request.width
    height = request.height

    # Split into 8 horizontal strips
    num_workers = 8
    strip_height = height // num_workers

    request_topic_path = pub.topic_path(PROJECT_ID, TOPIC_REQUESTS)

    with job_lock:
        jobs[job_id] = {
            "tiles": {},
            "total_tiles": num_workers,
            "future": asyncio.Future(),
            "loop": asyncio.get_event_loop(),
            "width": width,
            "height": height
        }

    try:
        futures = []
        for i in range(num_workers):
            start_y = i * strip_height
            current_h = strip_height
            if i == num_workers - 1:
                current_h = height - start_y

            payload = {
                "job_id": job_id,
                "scene": request.scene,
                "x": 0,
                "y": start_y,
                "w": width,
                "h": current_h,
                "full_w": width,
                "full_h": height,
                "samples": request.samples,
                "tile_index": i
            }

            data = json.dumps(payload).encode("utf-8")
            pub_future = pub.publish(request_topic_path, data)
            futures.append(pub_future)

        # Wait for job completion
        # We access the future directly from jobs, assuming it's still there.
        # It is safe because we haven't deleted it yet.
        await asyncio.wait_for(jobs[job_id]["future"], timeout=120.0)

        # Collect results
        with job_lock:
            job_data = jobs[job_id]
            sorted_tiles = []
            for i in range(num_workers):
                sorted_tiles.append(job_data["tiles"][i])

        return {
            "job_id": job_id,
            "tiles": sorted_tiles,
            "status": "completed"
        }

    except asyncio.TimeoutError:
        logger.error(f"Job {job_id} timed out")
        raise HTTPException(status_code=504, detail="Render timed out")
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        with job_lock:
            if job_id in jobs:
                del jobs[job_id]
