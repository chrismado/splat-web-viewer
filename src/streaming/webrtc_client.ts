/**
 * WebRTC Bi-Directional Streaming Client
 *
 * Receives compressed spherical harmonic delta updates from cloud GPU inference.
 * Enables sub-100ms interactive rendering of live neural network outputs.
 *
 * Architecture:
 *   Cloud GPU → compute SH coefficient deltas → compress → WebRTC datachannel
 *   Browser ← receive deltas → apply to local Gaussians → WebGPU rasterize
 *
 * Delta binary format (per packet):
 *   gaussianIndex: uint32   (4 bytes) — which Gaussian to update
 *   numCoeffs:     uint16   (2 bytes) — number of float32 coefficients
 *   coefficients:  float32[] (numCoeffs * 4 bytes) — changed SH coefficients
 *   timestamp:     float64  (8 bytes) — server-side timestamp for latency tracking
 */

export interface SphericalHarmonicDelta {
  gaussianIndex: number;
  coefficients: Float32Array;  // Only changed coefficients
  timestamp: number;
}

export class WebRTCSplatClient {
  private pc: RTCPeerConnection;
  private dataChannel: RTCDataChannel | null = null;
  private deltaCallback: ((delta: SphericalHarmonicDelta) => void) | null = null;

  constructor(private serverUrl: string) {
    this.pc = new RTCPeerConnection({
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
    });
  }

  /**
   * Connect to the signaling server via HTTP POST, perform full ICE negotiation.
   *
   * Signaling flow:
   *   1. Create data channel for SH deltas (unordered, no retransmit)
   *   2. Create SDP offer
   *   3. Set local description
   *   4. Gather ICE candidates
   *   5. POST offer to server, receive answer
   *   6. Set remote description
   */
  async connect(): Promise<void> {
    // Create the data channel for receiving SH delta updates
    this.dataChannel = this.pc.createDataChannel("sh_deltas", {
      ordered: false,       // UDP-like: prioritize latency over reliability
      maxRetransmits: 0,    // Never retransmit — stale deltas are useless
    });

    this.dataChannel.binaryType = "arraybuffer";

    this.dataChannel.onopen = () => {
      console.log("[WebRTC] Data channel opened");
    };

    this.dataChannel.onclose = () => {
      console.log("[WebRTC] Data channel closed");
    };

    this.dataChannel.onerror = (ev) => {
      console.error("[WebRTC] Data channel error:", ev);
    };

    // Wire up message handler if callback already registered
    this.dataChannel.onmessage = (event) => {
      if (this.deltaCallback && event.data instanceof ArrayBuffer) {
        const delta = decompressDelta(event.data);
        this.deltaCallback(delta);
      }
    };

    // Also handle server-initiated data channels (the server may create its own)
    this.pc.ondatachannel = (event) => {
      const channel = event.channel;
      if (channel.label === "sh_deltas") {
        channel.binaryType = "arraybuffer";
        this.dataChannel = channel;
        channel.onmessage = (msgEvent) => {
          if (this.deltaCallback && msgEvent.data instanceof ArrayBuffer) {
            const delta = decompressDelta(msgEvent.data);
            this.deltaCallback(delta);
          }
        };
      }
    };

    // Create SDP offer
    const offer = await this.pc.createOffer();
    await this.pc.setLocalDescription(offer);

    // Wait for ICE gathering to complete (or timeout after 5s)
    await this.waitForIceGathering(5000);

    // Send the offer (with gathered ICE candidates) to the signaling server
    const response = await fetch(`${this.serverUrl}/offer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sdp: this.pc.localDescription!.sdp,
        type: this.pc.localDescription!.type,
      }),
    });

    if (!response.ok) {
      throw new Error(`Signaling server returned ${response.status}: ${await response.text()}`);
    }

    const answer = await response.json();
    await this.pc.setRemoteDescription(
      new RTCSessionDescription({ sdp: answer.sdp, type: answer.type })
    );
  }

  /**
   * Wait for ICE gathering to complete or timeout.
   */
  private waitForIceGathering(timeoutMs: number): Promise<void> {
    return new Promise<void>((resolve) => {
      if (this.pc.iceGatheringState === "complete") {
        resolve();
        return;
      }

      const timeout = setTimeout(() => {
        resolve(); // Proceed with candidates gathered so far
      }, timeoutMs);

      this.pc.onicegatheringstatechange = () => {
        if (this.pc.iceGatheringState === "complete") {
          clearTimeout(timeout);
          resolve();
        }
      };
    });
  }

  /**
   * Register a callback for incoming SH delta updates.
   */
  onDeltaReceived(callback: (delta: SphericalHarmonicDelta) => void): void {
    this.deltaCallback = callback;

    // If channel is already open, wire up immediately
    if (this.dataChannel) {
      this.dataChannel.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
          const delta = decompressDelta(event.data);
          callback(delta);
        }
      };
    }
  }

  /**
   * Send a message to the server (e.g., camera pose updates for view-dependent streaming).
   */
  sendCameraPose(pose: { position: Float32Array; rotation: Float32Array }): void {
    if (!this.dataChannel || this.dataChannel.readyState !== "open") return;
    const buffer = new ArrayBuffer(28); // 3 floats position + 4 floats rotation
    const view = new Float32Array(buffer);
    view.set(pose.position, 0);
    view.set(pose.rotation, 3);
    this.dataChannel.send(buffer);
  }

  /**
   * Close the WebRTC connection.
   */
  disconnect(): void {
    if (this.dataChannel) {
      this.dataChannel.close();
      this.dataChannel = null;
    }
    this.pc.close();
  }
}

/**
 * Parse a binary delta packet from the WebRTC data channel.
 *
 * Binary layout:
 *   [0..3]   gaussianIndex: uint32 LE
 *   [4..5]   numCoeffs:     uint16 LE
 *   [6..6+numCoeffs*4-1] coefficients: float32[] LE
 *   [last 8 bytes]        timestamp: float64 LE
 */
function decompressDelta(buffer: ArrayBuffer): SphericalHarmonicDelta {
  const view = new DataView(buffer);
  const minSize = 4 + 2 + 8; // gaussianIndex + numCoeffs + timestamp

  if (buffer.byteLength < minSize) {
    throw new Error(`Delta packet too small: ${buffer.byteLength} bytes, minimum ${minSize}`);
  }

  let offset = 0;

  // Gaussian index (uint32)
  const gaussianIndex = view.getUint32(offset, true);
  offset += 4;

  // Number of coefficients (uint16)
  const numCoeffs = view.getUint16(offset, true);
  offset += 2;

  const expectedSize = 4 + 2 + numCoeffs * 4 + 8;
  if (buffer.byteLength < expectedSize) {
    throw new Error(
      `Delta packet truncated: ${buffer.byteLength} bytes, expected ${expectedSize}`
    );
  }

  // Coefficients (float32 array)
  const coefficients = new Float32Array(numCoeffs);
  for (let i = 0; i < numCoeffs; i++) {
    coefficients[i] = view.getFloat32(offset, true);
    offset += 4;
  }

  // Timestamp (float64)
  const timestamp = view.getFloat64(offset, true);

  return { gaussianIndex, coefficients, timestamp };
}
