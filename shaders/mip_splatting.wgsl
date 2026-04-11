// 2D Mip-Filter Fragment Shader for Anti-Aliased Gaussian Splatting
//
// Standard 3DGS uses a 2D dilation filter in screen space, causing:
//   - Popping artifacts when camera focal length changes
//   - High-frequency aliasing at extreme zooms
//   - Shrinkage bias causing degenerate Gaussians to exceed Nyquist limit
//
// This shader implements the 3D smoothing filter from Mip-Splatting (CVPR 2024):
//   1. 3D smoothing filter: constrains Gaussian frequency content based on
//      the maximal sampling frequency of training views. Replaces each 3D
//      Gaussian Σ with Σ' = Σ + s_max * I, where s_max is derived from the
//      coarsest training-view pixel footprint in world space.
//   2. 2D Mip filter: simulates a physical camera box filter in screen space.
//      Convolves the projected 2D Gaussian with a box the size of one pixel,
//      which adds (pixel_size²/12) * I to the 2D covariance.
//   Eliminates aliasing at any focal length or viewing distance.
//
// Reference: Mip-Splatting (Zehao Yu et al., CVPR 2024 Best Student Paper)

// Uniforms for camera and mip-filter parameters
struct MipUniforms {
    // Camera parameters
    viewport: vec2<f32>,         // viewport width, height in pixels
    focal: vec2<f32>,            // focal length fx, fy in pixels
    // 3D smoothing filter parameter
    // s_max: maximal sampling interval from training views (world-space units²)
    // Derived from coarsest training camera: s_max = (1 / (2 * f_train_min))²
    // where f_train_min is the minimum training-view focal length
    max_sampling_interval: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> uniforms: MipUniforms;

// Per-splat data passed from vertex shader
struct FragmentInput {
    @builtin(position) frag_coord: vec4<f32>,
    @location(0) color: vec4<f32>,
    // 2D Gaussian parameters in screen space (from 3D→2D projection)
    @location(1) conic: vec3<f32>,    // inverse 2D covariance: [a, b, c] where Σ⁻¹ = [[a, b],[b, c]]
    @location(2) center: vec2<f32>,   // 2D center in screen pixels
    @location(3) opacity: f32,        // pre-multiplied opacity (alpha after activation)
    // 3D covariance eigenvalues projected to screen, for 3D smoothing
    @location(4) cov2d_diag: vec2<f32>, // diagonal of 2D covariance [σ²_x, σ²_y] before mip
};

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    // ── Step 1: 3D Smoothing Filter ──
    // The 3D smoothing filter has already been applied to the 3D covariance
    // before projection (Σ_3d' = Σ_3d + s_max * I). The vertex shader handles
    // the 3D→2D projection with the smoothed covariance. The conic and cov2d_diag
    // arriving here already include the 3D smoothing contribution.

    // ── Step 2: 2D Mip Filter (Box Filter Convolution) ──
    // Convolve the 2D Gaussian with a box filter the size of one pixel.
    // A uniform box of width h has variance h²/12.
    // For a pixel, h_x = viewport.x / viewport.x = 1 pixel in normalized coords,
    // so in pixel coordinates: h = 1 pixel → variance = 1/12.
    // The convolved covariance: Σ_mip = Σ_2d + (1/12) * I
    //
    // We work with the inverse covariance (conic). Reconstruct Σ, add the
    // box-filter term, then re-invert.
    let a = in.conic.x;  // Σ⁻¹[0,0]
    let b = in.conic.y;  // Σ⁻¹[0,1] = Σ⁻¹[1,0]
    let c = in.conic.z;  // Σ⁻¹[1,1]

    // Invert the conic to recover Σ_2d
    let det_inv = a * c - b * b;
    if (det_inv <= 0.0) {
        discard;
    }
    let inv_det = 1.0 / det_inv;
    let sigma_xx = c * inv_det;
    let sigma_xy = -b * inv_det;
    let sigma_yy = a * inv_det;

    // Add box filter variance: Σ_mip = Σ_2d + (1/12) * I
    let box_var = 1.0 / 12.0;
    let mip_xx = sigma_xx + box_var;
    let mip_xy = sigma_xy;
    let mip_yy = sigma_yy + box_var;

    // Re-invert to get the mip-filtered conic: Σ_mip⁻¹
    let det_mip = mip_xx * mip_yy - mip_xy * mip_xy;
    if (det_mip <= 0.0) {
        discard;
    }
    let inv_det_mip = 1.0 / det_mip;
    let mip_a = mip_yy * inv_det_mip;
    let mip_b = -mip_xy * inv_det_mip;
    let mip_c = mip_xx * inv_det_mip;

    // ── Step 3: Evaluate the Mip-Filtered Gaussian ──
    // Compute the offset from Gaussian center to this fragment
    let d = in.frag_coord.xy - in.center;

    // Mahalanobis distance: d^T Σ_mip⁻¹ d
    let power = -0.5 * (mip_a * d.x * d.x + 2.0 * mip_b * d.x * d.y + mip_c * d.y * d.y);

    // Early exit if too far from center (> 3σ)
    if (power < -4.5) {  // exp(-4.5) ≈ 0.011, negligible contribution
        discard;
    }

    // ── Step 4: Normalization correction ──
    // The convolution changes the normalization: the peak of the convolved
    // Gaussian is scaled by det(Σ_2d) / det(Σ_mip) to conserve total energy.
    let norm_correction = sqrt(det_inv / det_mip);

    // Final Gaussian weight with mip-corrected opacity
    let gaussian = exp(power);
    let alpha = min(in.opacity * gaussian * norm_correction, 0.999);

    // Skip near-invisible fragments
    if (alpha < 1.0 / 255.0) {
        discard;
    }

    // Output pre-multiplied alpha color
    return vec4<f32>(in.color.rgb * alpha, alpha);
}

// Vertex shader: projects 3D Gaussians to 2D with 3D smoothing filter applied
struct VertexUniforms {
    view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    camera_pos: vec3<f32>,
    max_sampling_interval: f32,   // s_max for 3D smoothing
    viewport: vec2<f32>,
    focal: vec2<f32>,
};

@group(0) @binding(1) var<uniform> vert_uniforms: VertexUniforms;

struct SplatData {
    position: vec3<f32>,
    opacity: f32,
    color: vec4<f32>,
    // 3D covariance stored as upper-triangle: [Σ00, Σ01, Σ02, Σ11, Σ12, Σ22]
    cov3d: array<f32, 6>,
};

@group(0) @binding(2) var<storage, read> splats: array<SplatData>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) conic: vec3<f32>,
    @location(2) center: vec2<f32>,
    @location(3) opacity: f32,
    @location(4) cov2d_diag: vec2<f32>,
};

// Compute the 2D covariance from the 3D covariance via the Jacobian of the
// perspective projection (EWA splatting).
fn compute_cov2d(
    cov3d_0: f32, cov3d_1: f32, cov3d_2: f32,
    cov3d_3: f32, cov3d_4: f32, cov3d_5: f32,
    mean_view: vec3<f32>,
    focal: vec2<f32>,
    s_max: f32,
) -> mat2x2<f32> {
    // ── 3D Smoothing Filter ──
    // Σ'_3d = Σ_3d + s_max * I
    // This constrains the maximal frequency of the Gaussian to be below
    // the Nyquist limit of the coarsest training view.
    let sm0 = cov3d_0 + s_max;
    let sm1 = cov3d_1;
    let sm2 = cov3d_2;
    let sm3 = cov3d_3 + s_max;
    let sm4 = cov3d_4;
    let sm5 = cov3d_5 + s_max;

    // Jacobian of the perspective projection
    let tz = mean_view.z;
    let tz2 = tz * tz;
    let tx = mean_view.x;
    let ty = mean_view.y;

    let J = mat3x2<f32>(
        vec2<f32>(focal.x / tz, 0.0),
        vec2<f32>(0.0, focal.y / tz),
        vec2<f32>(-focal.x * tx / tz2, -focal.y * ty / tz2),
    );

    // Σ_2d = J * Σ'_3d * J^T
    // Expand manually for the symmetric 3x3 cov matrix
    // First compute T = Σ'_3d * J^T (3x2)
    let t00 = sm0 * J[0].x + sm1 * J[1].x + sm2 * J[2].x;
    let t01 = sm0 * J[0].y + sm1 * J[1].y + sm2 * J[2].y;
    let t10 = sm1 * J[0].x + sm3 * J[1].x + sm4 * J[2].x;
    let t11 = sm1 * J[0].y + sm3 * J[1].y + sm4 * J[2].y;
    let t20 = sm2 * J[0].x + sm4 * J[1].x + sm5 * J[2].x;
    let t21 = sm2 * J[0].y + sm4 * J[1].y + sm5 * J[2].y;

    // Then Σ_2d = J * T
    let cov2d_00 = J[0].x * t00 + J[1].x * t10 + J[2].x * t20;
    let cov2d_01 = J[0].x * t01 + J[1].x * t11 + J[2].x * t21;
    let cov2d_11 = J[0].y * t01 + J[1].y * t11 + J[2].y * t21;

    return mat2x2<f32>(
        vec2<f32>(cov2d_00, cov2d_01),
        vec2<f32>(cov2d_01, cov2d_11),
    );
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_idx: u32,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    var out: VertexOutput;

    let splat = splats[instance_idx];

    // Transform to view space. The demo camera follows WebGL convention:
    // visible geometry sits in front of the camera along negative Z.
    let pos_world = vec4<f32>(splat.position, 1.0);
    let pos_view = vert_uniforms.view_matrix * pos_world;

    // Skip splats behind camera
    if (pos_view.z >= -0.2) {
        out.position = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        out.opacity = 0.0;
        return out;
    }

    // Compute 2D covariance with 3D smoothing filter applied
    let cov2d = compute_cov2d(
        splat.cov3d[0], splat.cov3d[1], splat.cov3d[2],
        splat.cov3d[3], splat.cov3d[4], splat.cov3d[5],
        vec3<f32>(pos_view.x, pos_view.y, -pos_view.z),
        vert_uniforms.focal,
        vert_uniforms.max_sampling_interval,
    );

    // Store diagonal for fragment shader normalization
    out.cov2d_diag = vec2<f32>(cov2d[0].x, cov2d[1].y);

    // Compute inverse covariance (conic)
    let det = cov2d[0].x * cov2d[1].y - cov2d[0].y * cov2d[1].x;
    if (det <= 0.0) {
        out.position = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        out.opacity = 0.0;
        return out;
    }
    let inv_det = 1.0 / det;
    out.conic = vec3<f32>(
        cov2d[1].y * inv_det,
        -cov2d[0].y * inv_det,
        cov2d[0].x * inv_det,
    );

    // Project center to screen
    let pos_clip = vert_uniforms.proj_matrix * pos_view;
    let ndc = pos_clip.xy / pos_clip.w;
    let screen_center = vec2<f32>(
        (ndc.x * 0.5 + 0.5) * vert_uniforms.viewport.x,
        (ndc.y * -0.5 + 0.5) * vert_uniforms.viewport.y,
    );
    out.center = screen_center;

    // Compute quad extent from eigenvalues of 2D covariance (3σ radius)
    let mid = 0.5 * (cov2d[0].x + cov2d[1].y);
    let disc = sqrt(max(0.1, mid * mid - det));
    let lambda1 = mid + disc;
    let lambda2 = mid - disc;
    let radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));

    // Emit a screen-space quad (4 vertices per splat, triangle strip)
    let quad_offsets = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
    );
    let quad_offset = quad_offsets[vertex_idx] * radius;
    let screen_pos = screen_center + quad_offset;

    // Convert back to clip space
    out.position = vec4<f32>(
        (screen_pos.x / vert_uniforms.viewport.x) * 2.0 - 1.0,
        -((screen_pos.y / vert_uniforms.viewport.y) * 2.0 - 1.0),
        pos_clip.z / pos_clip.w,
        1.0,
    );

    out.color = splat.color;
    out.opacity = splat.opacity;

    return out;
}
