/**
 * WebGPU Gaussian Splat Rasterizer
 *
 * Takes decoded GaussianSplat[] and renders them using WebGPU with the
 * mip_splatting.wgsl shader for anti-aliased rendering.
 *
 * Supports incremental SH delta updates from the WebRTC streaming client,
 * refreshing the sorted GPU buffer on the next render.
 */

import { GaussianSplat } from "../compression/spz_decoder";
import { SphericalHarmonicDelta } from "../streaming/webrtc_client";
import { cameraSortSignature, sortSplatsBackToFront } from "./splat_sort";

const MIP_UNIFORM_SIZE = 32;
const VERTEX_UNIFORM_SIZE = 160;

// Matches the SplatData struct in mip_splatting.wgsl. WGSL rounds the struct
// stride up to the largest member alignment, so each array element is 64 bytes.
const SPLAT_STRIDE = 16 * 4;

interface Camera {
  viewMatrix: Float32Array;  // 4x4 column-major
  projMatrix: Float32Array;  // 4x4 column-major
  position: Float32Array;    // [x, y, z]
  focalX: number;            // focal length X in pixels
  focalY: number;            // focal length Y in pixels
}

export class WebGPURasterizer {
  private device!: GPUDevice;
  private context!: GPUCanvasContext;
  private pipeline!: GPURenderPipeline;
  private uniformBuffer!: GPUBuffer;
  private vertexUniformBuffer!: GPUBuffer;
  private splatBuffer!: GPUBuffer;
  private bindGroup!: GPUBindGroup;
  private numSplats = 0;
  private canvas: HTMLCanvasElement;
  private maxSamplingInterval = 0.0003; // Default s_max for 3D smoothing filter
  private splats: GaussianSplat[] = [];
  private lastSortSignature = "";

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
  }

  /**
   * Initialize WebGPU device, pipeline, and buffers.
   */
  async initialize(): Promise<void> {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) {
      throw new Error("WebGPU not supported: no adapter found");
    }

    this.device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: 256 * 1024 * 1024, // 256MB for large scenes
      },
    });

    this.context = this.canvas.getContext("webgpu") as GPUCanvasContext;
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format,
      alphaMode: "premultiplied",
    });

    // Load the mip-splatting shader
    const shaderCode = await this.loadShader();
    const shaderModule = this.device.createShaderModule({ code: shaderCode });

    // Fragment mip-filter params:
    // viewport (8) + focal (8) + max_sampling_interval (4) + padding (12) = 32 bytes.
    this.uniformBuffer = this.device.createBuffer({
      size: MIP_UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // VertexUniforms:
    // view/proj matrices (128) + camera_pos/max_sampling_interval (16) + viewport/focal (16).
    this.vertexUniformBuffer = this.device.createBuffer({
      size: VERTEX_UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Create pipeline with alpha blending for Gaussian accumulation
    this.pipeline = this.device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: shaderModule,
        entryPoint: "vs_main",
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fs_main",
        targets: [{
          format,
          blend: {
            color: {
              srcFactor: "one",
              dstFactor: "one-minus-src-alpha",
              operation: "add",
            },
            alpha: {
              srcFactor: "one",
              dstFactor: "one-minus-src-alpha",
              operation: "add",
            },
          },
        }],
      },
      primitive: {
        topology: "triangle-strip",
      },
    });
  }

  /**
   * Upload decoded Gaussian splats to the GPU.
   */
  uploadSplats(splats: GaussianSplat[]): void {
    this.splats = splats.slice();
    this.numSplats = this.splats.length;
    if (this.numSplats === 0) return;

    const bufferSize = this.numSplats * SPLAT_STRIDE;

    // Destroy old buffer if it exists and is too small
    if (this.splatBuffer) {
      this.splatBuffer.destroy();
    }

    this.splatBuffer = this.device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.lastSortSignature = "";
    this.writeSplatsToBuffer(this.splats);

    // Create bind group with the new buffer
    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },  // MipUniforms
        { binding: 1, resource: { buffer: this.vertexUniformBuffer } },  // VertexUniforms
        { binding: 2, resource: { buffer: this.splatBuffer } },
      ],
    });
  }

  /**
   * Update the color of a specific Gaussian splat from new SH coefficients.
   * Marks the sorted GPU buffer dirty so the new color is uploaded next frame.
   */
  updateColors(delta: SphericalHarmonicDelta): void {
    if (!this.splatBuffer || delta.gaussianIndex >= this.numSplats) return;

    // SH coefficients affect the color of the Gaussian.
    // For real-time updates, we write new color values directly to the buffer.
    // The SH evaluation: color = SH_0 * c_dc + sum(SH_l_m * c_l_m)
    // Updates replace the existing c_l_m coefficients.
    //
    // For the base color (DC component, degree 0), the first 3 coefficients
    // map directly to RGB. We update the color field in the splat buffer.
    if (delta.coefficients.length >= 3) {
      const splat = this.splats[delta.gaussianIndex];
      splat.color[0] = delta.coefficients[0];
      splat.color[1] = delta.coefficients[1];
      splat.color[2] = delta.coefficients[2];
      this.lastSortSignature = "";
    }
  }

  /**
   * Render one frame with the given camera parameters.
   */
  render(camera: Camera): void {
    if (this.numSplats === 0 || !this.bindGroup) return;

    this.resortSplats(camera.position);

    // Update fragment mip-filter uniforms.
    const mipUniformData = new ArrayBuffer(MIP_UNIFORM_SIZE);
    const mipF32 = new Float32Array(mipUniformData);
    mipF32[0] = this.canvas.width;
    mipF32[1] = this.canvas.height;
    mipF32[2] = camera.focalX;
    mipF32[3] = camera.focalY;
    mipF32[4] = this.maxSamplingInterval;

    this.device.queue.writeBuffer(this.uniformBuffer, 0, mipUniformData);

    // Update vertex camera uniforms.
    const vertexUniformData = new ArrayBuffer(VERTEX_UNIFORM_SIZE);
    const f32 = new Float32Array(vertexUniformData);

    // view_matrix: mat4x4 at offset 0
    f32.set(camera.viewMatrix, 0);
    // proj_matrix: mat4x4 at offset 16
    f32.set(camera.projMatrix, 16);
    // camera_pos: vec3 at offset 32
    f32.set(camera.position, 32);
    // max_sampling_interval: f32 at offset 35
    f32[35] = this.maxSamplingInterval;
    // viewport: vec2 at offset 36
    f32[36] = this.canvas.width;
    f32[37] = this.canvas.height;
    // focal: vec2 at offset 38
    f32[38] = camera.focalX;
    f32[39] = camera.focalY;

    this.device.queue.writeBuffer(this.vertexUniformBuffer, 0, vertexUniformData);

    const encoder = this.device.createCommandEncoder();
    const textureView = this.context.getCurrentTexture().createView();

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: textureView,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    // Draw 4 vertices per splat (triangle strip quad), instanced over all splats
    pass.draw(4, this.numSplats, 0, 0);
    pass.end();

    this.device.queue.submit([encoder.finish()]);
  }

  /**
   * Set the 3D smoothing filter parameter.
   * s_max = (1 / (2 * f_train_min))² where f_train_min is the minimum
   * training-view focal length in world-space units per pixel.
   */
  setMaxSamplingInterval(sMax: number): void {
    this.maxSamplingInterval = sMax;
  }

  private writeSplatsToBuffer(splats: GaussianSplat[]): void {
    const data = new Float32Array(splats.length * (SPLAT_STRIDE / 4));
    const floatsPerSplat = SPLAT_STRIDE / 4;

    for (let i = 0; i < splats.length; i++) {
      const splat = splats[i];
      const base = i * floatsPerSplat;

      data[base + 0] = splat.position[0];
      data[base + 1] = splat.position[1];
      data[base + 2] = splat.position[2];
      data[base + 3] = splat.alpha;
      data[base + 4] = splat.color[0];
      data[base + 5] = splat.color[1];
      data[base + 6] = splat.color[2];
      data[base + 7] = 1.0;

      const cov3d = computeCov3D(splat.scale, splat.rotation);
      data[base + 8] = cov3d[0];
      data[base + 9] = cov3d[1];
      data[base + 10] = cov3d[2];
      data[base + 11] = cov3d[3];
      data[base + 12] = cov3d[4];
      data[base + 13] = cov3d[5];
    }

    this.device.queue.writeBuffer(this.splatBuffer, 0, data);
  }

  private resortSplats(cameraPosition: Float32Array): void {
    const signature = cameraSortSignature(cameraPosition);
    if (signature === this.lastSortSignature) {
      return;
    }

    const sortedSplats = sortSplatsBackToFront(this.splats, cameraPosition);
    this.writeSplatsToBuffer(sortedSplats);
    this.lastSortSignature = signature;
  }

  /**
   * Load the mip-splatting WGSL shader source.
   */
  private async loadShader(): Promise<string> {
    const candidates = [
      "shaders/mip_splatting.wgsl",
      "../shaders/mip_splatting.wgsl",
      "/shaders/mip_splatting.wgsl",
    ];

    for (const path of candidates) {
      const response = await fetch(path);
      if (response.ok) {
        return response.text();
      }
    }

    throw new Error("Failed to load shader from known demo paths");
  }

  /**
   * Clean up GPU resources.
   */
  destroy(): void {
    this.splatBuffer?.destroy();
    this.uniformBuffer?.destroy();
    this.vertexUniformBuffer?.destroy();
  }
}

/**
 * Compute the 3D covariance matrix from scale and rotation quaternion.
 *
 * Σ = R * S * S^T * R^T where S = diag(scale) and R = rotation matrix from quaternion.
 *
 * Returns upper-triangle: [Σ00, Σ01, Σ02, Σ11, Σ12, Σ22]
 */
function computeCov3D(scale: Float32Array, rotation: Float32Array): Float32Array {
  const [w, x, y, z] = rotation;

  // Rotation matrix from quaternion
  const r00 = 1 - 2 * (y * y + z * z);
  const r01 = 2 * (x * y - w * z);
  const r02 = 2 * (x * z + w * y);
  const r10 = 2 * (x * y + w * z);
  const r11 = 1 - 2 * (x * x + z * z);
  const r12 = 2 * (y * z - w * x);
  const r20 = 2 * (x * z - w * y);
  const r21 = 2 * (y * z + w * x);
  const r22 = 1 - 2 * (x * x + y * y);

  const sx = scale[0], sy = scale[1], sz = scale[2];
  const sx2 = sx * sx, sy2 = sy * sy, sz2 = sz * sz;

  // M = R * S (columns of R scaled by S)
  // Σ = M * M^T
  const cov00 = r00 * r00 * sx2 + r01 * r01 * sy2 + r02 * r02 * sz2;
  const cov01 = r00 * r10 * sx2 + r01 * r11 * sy2 + r02 * r12 * sz2;
  const cov02 = r00 * r20 * sx2 + r01 * r21 * sy2 + r02 * r22 * sz2;
  const cov11 = r10 * r10 * sx2 + r11 * r11 * sy2 + r12 * r12 * sz2;
  const cov12 = r10 * r20 * sx2 + r11 * r21 * sy2 + r12 * r22 * sz2;
  const cov22 = r20 * r20 * sx2 + r21 * r21 * sy2 + r22 * r22 * sz2;

  return new Float32Array([cov00, cov01, cov02, cov11, cov12, cov22]);
}
