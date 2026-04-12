"""
WebRTC Signaling + SH Delta Streaming Server

Runs on the cloud GPU side. Accepts WebRTC connections from the browser client,
computes spherical harmonic coefficient deltas from a neural network model,
and streams them over a WebRTC data channel at sub-100ms latency.

Delta binary format (matches webrtc_client.ts):
    gaussianIndex: uint32 LE   (4 bytes)
    numCoeffs:     uint16 LE   (2 bytes)
    coefficients:  float32[] LE (numCoeffs * 4 bytes)
    timestamp:     float64 LE  (8 bytes)

Usage:
    python streaming/webrtc_server.py --host 0.0.0.0 --port 8081
"""

import argparse
import asyncio
import logging
import struct
import time

import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel

logger = logging.getLogger("webrtc_server")

# Track active peer connections for cleanup
peer_connections: set[RTCPeerConnection] = set()


def encode_sh_delta(
    gaussian_index: int,
    coefficients: np.ndarray,
    timestamp: float,
) -> bytes:
    """
    Encode a spherical harmonic delta update into the binary wire format.

    Args:
        gaussian_index: Index of the Gaussian to update.
        coefficients: Changed SH coefficients as float32 array.
        timestamp: Server-side timestamp for latency measurement.

    Returns:
        Binary packet matching the format expected by decompressDelta() in
        webrtc_client.ts.
    """
    coefficients = coefficients.astype(np.float32)
    num_coeffs = len(coefficients)

    # Pack: uint32 + uint16 + float32[numCoeffs] + float64
    header = struct.pack("<IH", gaussian_index, num_coeffs)
    coeff_bytes = coefficients.tobytes()
    timestamp_bytes = struct.pack("<d", timestamp)

    return header + coeff_bytes + timestamp_bytes


async def send_sh_deltas(
    channel: RTCDataChannel,
    num_gaussians: int = 100_000,
    target_fps: int = 30,
) -> None:
    """
    Continuously compute and send SH coefficient deltas over the data channel.

    In production, this would run neural network inference on the GPU to
    compute which Gaussians changed and by how much. For now, we simulate
    the delta computation to validate the streaming pipeline.

    Args:
        channel: Open WebRTC data channel to send deltas on.
        num_gaussians: Total number of Gaussians in the scene.
        target_fps: Target update rate in frames per second.
    """
    interval = 1.0 / target_fps
    frame = 0

    while channel.readyState == "open":
        t_start = time.monotonic()
        timestamp = time.time()

        # Simulation: send small random deltas for a subset of Gaussians.
        # The real inference-backed SH delta path is not integrated yet, so the
        # server is intentionally explicit about being a streaming harness.
        num_changed = max(1, num_gaussians // 100)
        changed_indices = np.random.choice(
            num_gaussians, size=num_changed, replace=False
        )

        for idx in changed_indices:
            # Simulate small SH coefficient changes (3 RGB DC coefficients)
            coeffs = np.random.randn(3).astype(np.float32) * 0.001
            packet = encode_sh_delta(int(idx), coeffs, timestamp)

            try:
                channel.send(packet)
            except Exception as e:
                logger.warning("Failed to send delta: %s", e)
                return

        frame += 1

        # Pace to target FPS
        elapsed = time.monotonic() - t_start
        sleep_time = interval - elapsed
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)


async def handle_offer(request: web.Request) -> web.Response:
    """
    HTTP POST /offer — WebRTC signaling endpoint.

    Receives an SDP offer from the browser, creates an answer, and sets up
    the data channel for SH delta streaming.
    """
    body = await request.json()
    offer = RTCSessionDescription(sdp=body["sdp"], type=body["type"])

    pc = RTCPeerConnection()
    peer_connections.add(pc)

    @pc.on("connectionstatechange")
    async def on_connection_state_change() -> None:
        logger.info("Connection state: %s", pc.connectionState)
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            peer_connections.discard(pc)

    @pc.on("datachannel")
    def on_datachannel(channel: RTCDataChannel) -> None:
        logger.info("Data channel opened: %s", channel.label)

        if channel.label == "sh_deltas":
            # Start streaming SH deltas on this channel
            asyncio.ensure_future(send_sh_deltas(channel))

        @channel.on("message")
        def on_message(message: bytes) -> None:
            # Receive camera pose updates from the browser
            if isinstance(message, bytes) and len(message) == 28:
                # 7 floats: position(3) + rotation quaternion(4)
                pose = struct.unpack("<7f", message)
                logger.debug("Camera pose: pos=%s rot=%s", pose[:3], pose[3:])

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    })


async def on_shutdown(app: web.Application) -> None:
    """Close all peer connections on server shutdown."""
    coros = [pc.close() for pc in peer_connections]
    await asyncio.gather(*coros)
    peer_connections.clear()


def create_app() -> web.Application:
    app = web.Application()

    # CORS headers for browser access
    async def cors_middleware(
        request: web.Request,
        handler,
    ) -> web.Response:
        if request.method == "OPTIONS":
            response = web.Response()
        else:
            response = await handler(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    app.middlewares.append(cors_middleware)
    app.router.add_post("/offer", handle_offer)
    app.on_shutdown.append(on_shutdown)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="WebRTC SH delta streaming harness")
    parser.add_argument("--host", default="0.0.0.0", help="Listen address")
    parser.add_argument("--port", type=int, default=8081, help="Listen port")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logger.info("Starting WebRTC server on %s:%d", args.host, args.port)

    app = create_app()
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
