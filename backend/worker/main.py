import os
import json
import base64
import numpy as np
import logging
from google.cloud import pubsub_v1
import sys

# Ensure we can import the renderer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

try:
    from avantime_rat_tracer import render_tile
except ImportError:
    logging.error("Could not import avantime_rat_tracer. Make sure you are running from the correct directory or have installed the package.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "test-project")
TOPIC_RESULTS = "render-results"
SUBSCRIPTION_REQUESTS = "render-requests-sub"

# Initialize clients lazily or handle errors if environment not set
try:
    publisher = pubsub_v1.PublisherClient()
    subscriber = pubsub_v1.SubscriberClient()
except Exception as e:
    logger.warning(f"Could not init PubSub clients: {e}")
    publisher = None
    subscriber = None

def callback(message):
    try:
        logger.info("Received render request")
        data = json.loads(message.data.decode("utf-8"))

        job_id = data.get("job_id")
        scene = data.get("scene")
        x = data.get("x")
        y = data.get("y")
        w = data.get("w")
        h = data.get("h")
        full_w = data.get("full_w")
        full_h = data.get("full_h")
        samples = data.get("samples", 1024)
        tile_index = data.get("tile_index")

        message.ack()

        # Render
        # render_tile returns a JAX array (numpy compatible)
        # shape (h, w, 3), float32
        result_array = render_tile(scene, x, y, w, h, full_w, full_h, samples)

        # Convert to numpy and then to uint8 for transmission
        np_result = np.array(result_array)
        img_uint8 = (np.clip(np_result, 0, 1) * 255).astype(np.uint8)

        # Serialize to base64
        # We send raw bytes of the pixel data
        result_bytes = img_uint8.tobytes()
        result_b64 = base64.b64encode(result_bytes).decode("utf-8")

        # Publish result
        response = {
            "job_id": job_id,
            "tile_index": tile_index,
            "result": result_b64,
            "shape": list(img_uint8.shape),
            "dtype": str(img_uint8.dtype)
        }

        if publisher:
            result_topic_path = publisher.topic_path(PROJECT_ID, TOPIC_RESULTS)
            publisher.publish(result_topic_path, json.dumps(response).encode("utf-8"))
            logger.info(f"Published result for job {job_id}, tile {tile_index}")
        else:
            logger.error("Publisher not initialized")

    except Exception as e:
        logger.error(f"Error processing render request: {e}")
        message.nack()

def main():
    if not subscriber:
        logger.error("Subscriber not initialized")
        return

    request_subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_REQUESTS)
    streaming_pull_future = subscriber.subscribe(request_subscription_path, callback=callback)
    logger.info(f"Worker listening on {request_subscription_path}")

    with subscriber:
        try:
            streaming_pull_future.result()
        except KeyboardInterrupt:
            streaming_pull_future.cancel()
        except Exception as e:
            logger.error(f"Worker failed: {e}")

if __name__ == "__main__":
    main()
