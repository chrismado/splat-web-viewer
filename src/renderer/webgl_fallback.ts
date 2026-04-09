/**
 * WebGL 2.0 Fallback Renderer
 *
 * Provides Gaussian splat rendering on devices without WebGPU support.
 * Uses instanced rendering of camera-facing quads with alpha blending.
 *
 * Limitations vs WebGPU renderer:
 *   - No compute-shader-based sorting (uses CPU front-to-back sort)
 *   - No mip-splatting anti-aliasing (uses basic 2D Gaussian falloff)
 *   - Lower max splat count (~500k vs ~2M on WebGPU)
 */

import { GaussianSplat } from "../compression/spz_decoder";

const VERT_SHADER = `#version 300 es
precision highp float;

// Per-instance attributes
layout(location = 0) in vec3 a_center;    // Gaussian center (world)
layout(location = 1) in float a_opacity;
layout(location = 2) in vec3 a_color;
layout(location = 3) in vec3 a_scale;
layout(location = 4) in vec4 a_rotation;  // quaternion [w, x, y, z]

// Per-vertex quad corner: [-1,-1], [1,-1], [-1,1], [1,1]
layout(location = 5) in vec2 a_quad;

uniform mat4 u_viewMatrix;
uniform mat4 u_projMatrix;
uniform vec2 u_viewport;
uniform vec2 u_focal;

out vec2 v_uv;
out vec3 v_color;
out float v_opacity;

mat3 quatToMat3(vec4 q) {
  float w = q.x, x = q.y, y = q.z, z = q.w;
  return mat3(
    1.0 - 2.0*(y*y + z*z), 2.0*(x*y + w*z),       2.0*(x*z - w*y),
    2.0*(x*y - w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z + w*x),
    2.0*(x*z + w*y),       2.0*(y*z - w*x),       1.0 - 2.0*(x*x + y*y)
  );
}

void main() {
  mat3 R = quatToMat3(a_rotation);
  mat3 S = mat3(a_scale.x, 0.0, 0.0,
                0.0, a_scale.y, 0.0,
                0.0, 0.0, a_scale.z);
  mat3 M = R * S;
  mat3 Sigma = M * transpose(M);

  vec4 viewPos = u_viewMatrix * vec4(a_center, 1.0);
  float depth = -viewPos.z;
  if (depth < 0.1) {
    gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
    return;
  }

  // Project covariance to 2D
  mat3 V3 = mat3(u_viewMatrix);
  mat3 cov2d_full = V3 * Sigma * transpose(V3);
  // Extract 2x2 from the XY block
  float a = cov2d_full[0][0];
  float b = cov2d_full[0][1];
  float d = cov2d_full[1][1];

  // Eigenvalues for the 2D extent
  float det = a * d - b * b;
  float trace = a + d;
  float gap = max(0.0, trace * trace * 0.25 - det);
  float sqrtGap = sqrt(gap);
  float lambda1 = max(trace * 0.5 + sqrtGap, 0.0001);
  float lambda2 = max(trace * 0.5 - sqrtGap, 0.0001);
  float radius = 3.0 * sqrt(max(lambda1, lambda2));

  vec2 screenScale = radius * u_focal / depth;
  vec2 quadOffset = a_quad * screenScale / u_viewport;

  vec4 clipPos = u_projMatrix * viewPos;
  clipPos.xy += quadOffset * clipPos.w;

  gl_Position = clipPos;
  v_uv = a_quad;
  v_color = a_color;
  v_opacity = a_opacity;
}
`;

const FRAG_SHADER = `#version 300 es
precision highp float;

in vec2 v_uv;
in vec3 v_color;
in float v_opacity;

out vec4 fragColor;

void main() {
  float d2 = dot(v_uv, v_uv);
  // Gaussian falloff: exp(-d^2 / 2), clipped at 3 sigma
  float alpha = v_opacity * exp(-0.5 * d2);
  if (alpha < 1.0 / 255.0) discard;
  fragColor = vec4(v_color * alpha, alpha);
}
`;

export class WebGLFallbackRenderer {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram | null = null;
  private vao: WebGLVertexArrayObject | null = null;
  private instanceBuffer: WebGLBuffer | null = null;
  private numSplats = 0;
  private canvas: HTMLCanvasElement;

  // Uniform locations
  private uViewMatrix: WebGLUniformLocation | null = null;
  private uProjMatrix: WebGLUniformLocation | null = null;
  private uViewport: WebGLUniformLocation | null = null;
  private uFocal: WebGLUniformLocation | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", {
      alpha: true,
      premultipliedAlpha: true,
      antialias: false,
    });
    if (!gl) {
      throw new Error("WebGL 2 not available");
    }
    this.gl = gl;
  }

  initialize(): void {
    const gl = this.gl;

    // Compile shaders
    const vs = this.compileShader(gl.VERTEX_SHADER, VERT_SHADER);
    const fs = this.compileShader(gl.FRAGMENT_SHADER, FRAG_SHADER);
    this.program = gl.createProgram()!;
    gl.attachShader(this.program, vs);
    gl.attachShader(this.program, fs);
    gl.linkProgram(this.program);

    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      throw new Error(`Shader link error: ${gl.getProgramInfoLog(this.program)}`);
    }

    this.uViewMatrix = gl.getUniformLocation(this.program, "u_viewMatrix");
    this.uProjMatrix = gl.getUniformLocation(this.program, "u_projMatrix");
    this.uViewport = gl.getUniformLocation(this.program, "u_viewport");
    this.uFocal = gl.getUniformLocation(this.program, "u_focal");

    // Setup VAO
    this.vao = gl.createVertexArray()!;
    gl.bindVertexArray(this.vao);

    // Quad vertices (triangle strip)
    const quadVerts = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    const quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, quadVerts, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(5);
    gl.vertexAttribPointer(5, 2, gl.FLOAT, false, 0, 0);

    // Instance buffer (will be populated in uploadSplats)
    this.instanceBuffer = gl.createBuffer()!;

    gl.bindVertexArray(null);

    // Blending for Gaussian accumulation
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.disable(gl.DEPTH_TEST);
  }

  uploadSplats(splats: GaussianSplat[]): void {
    this.numSplats = splats.length;
    if (this.numSplats === 0) return;

    const gl = this.gl;

    // Pack instance data: center(3) + opacity(1) + color(3) + scale(3) + rotation(4) = 14 floats
    const floatsPerInstance = 14;
    const data = new Float32Array(this.numSplats * floatsPerInstance);

    for (let i = 0; i < this.numSplats; i++) {
      const s = splats[i];
      const base = i * floatsPerInstance;
      data[base + 0] = s.position[0];
      data[base + 1] = s.position[1];
      data[base + 2] = s.position[2];
      data[base + 3] = s.alpha;
      data[base + 4] = s.color[0];
      data[base + 5] = s.color[1];
      data[base + 6] = s.color[2];
      data[base + 7] = s.scale[0];
      data[base + 8] = s.scale[1];
      data[base + 9] = s.scale[2];
      data[base + 10] = s.rotation[0];
      data[base + 11] = s.rotation[1];
      data[base + 12] = s.rotation[2];
      data[base + 13] = s.rotation[3];
    }

    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer!);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);

    const stride = floatsPerInstance * 4;
    // a_center (location 0)
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, stride, 0);
    gl.vertexAttribDivisor(0, 1);
    // a_opacity (location 1)
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 1, gl.FLOAT, false, stride, 12);
    gl.vertexAttribDivisor(1, 1);
    // a_color (location 2)
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 3, gl.FLOAT, false, stride, 16);
    gl.vertexAttribDivisor(2, 1);
    // a_scale (location 3)
    gl.enableVertexAttribArray(3);
    gl.vertexAttribPointer(3, 3, gl.FLOAT, false, stride, 28);
    gl.vertexAttribDivisor(3, 1);
    // a_rotation (location 4)
    gl.enableVertexAttribArray(4);
    gl.vertexAttribPointer(4, 4, gl.FLOAT, false, stride, 40);
    gl.vertexAttribDivisor(4, 1);

    gl.bindVertexArray(null);
  }

  render(camera: {
    viewMatrix: Float32Array;
    projMatrix: Float32Array;
    focalX: number;
    focalY: number;
  }): void {
    if (this.numSplats === 0 || !this.program) return;

    const gl = this.gl;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.uViewMatrix, false, camera.viewMatrix);
    gl.uniformMatrix4fv(this.uProjMatrix, false, camera.projMatrix);
    gl.uniform2f(this.uViewport, this.canvas.width, this.canvas.height);
    gl.uniform2f(this.uFocal, camera.focalX, camera.focalY);

    gl.bindVertexArray(this.vao);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.numSplats);
    gl.bindVertexArray(null);
  }

  destroy(): void {
    const gl = this.gl;
    if (this.program) gl.deleteProgram(this.program);
    if (this.vao) gl.deleteVertexArray(this.vao);
    if (this.instanceBuffer) gl.deleteBuffer(this.instanceBuffer);
  }

  private compileShader(type: number, source: string): WebGLShader {
    const gl = this.gl;
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`Shader compile error: ${info}`);
    }
    return shader;
  }
}
