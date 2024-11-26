#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{Vertex, VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{Vertex, VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif
#import bevy_pbr::{
    utils::rand_f,
    mesh_bindings::mesh,
    mesh_functions,
    mesh_view_bindings::globals,
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    view_transformations::position_world_to_clip
}

// from https://bottosson.github.io/posts/oklab/
fn linear_srgb_to_oklab(c: vec3<f32>) -> vec3<f32> {
    let l = 0.4122214708  * c.r + 0.5363325363  * c.g + 0.0514459929  * c.b;
	let m = 0.2119034982  * c.r + 0.6806995451  * c.g + 0.1073969566  * c.b;
	let s = 0.0883024619  * c.r + 0.2817188376  * c.g + 0.6299787005  * c.b;

    let l_ = pow(l, 1.0 / 3.0);
    let m_ = pow(m, 1.0 / 3.0);
    let s_ = pow(s, 1.0 / 3.0);

    return vec3(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    );
}

fn oklab_to_linear_srgb(c: vec3<f32>) -> vec3<f32> {
    let l_ = c.x + 0.3963377774 * c.y + 0.2158037573 * c.z;
    let m_ = c.x - 0.1055613458 * c.y - 0.0638541728 * c.z;
    let s_ = c.x - 0.0894841775 * c.y - 1.2914855480 * c.z;

    let l = l_*l_*l_;
    let m = m_*m_*m_;
    let s = s_*s_*s_;

    return vec3(
		4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
		-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
		-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    );
}

struct ScribbleOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
#ifdef VERTEX_UVS_A
    @location(2) uv: vec2<f32>,
#endif
#ifdef VERTEX_UVS_B
    @location(3) uv_b: vec2<f32>,
#endif
#ifdef VERTEX_TANGENTS
    @location(4) world_tangent: vec4<f32>,
#endif
#ifdef VERTEX_COLORS
    @location(5) color: vec4<f32>,
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    @location(6) @interpolate(flat) instance_index: u32,
#endif
#ifdef VISIBILITY_RANGE_DITHER
    @location(7) @interpolate(flat) visibility_range_dither: i32,
#endif
    @location(8) no_scribble_pos: vec2<f32>
}

fn vertex_from_scribble(scribble: ScribbleOutput) -> VertexOutput {
    var out: VertexOutput;
    out.position = scribble.position;
    out.world_position = scribble.world_position;
    out.world_normal = scribble.world_normal;
#ifdef VERTEX_UVS_A
    out.uv = scribble.uv;
#endif
#ifdef VERTEX_UVS_B
    out.uv_b = scribble.uv_b;
#endif
#ifdef VERTEX_TANGENTS
    out.world_tangent = scribble.world_tangent;
#endif
#ifdef VERTEX_COLORS
    out.color = scribble.color;
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = scribble.instance_index;
#endif
#ifdef VISIBILITY_RANGE_DITHER
    out.visibility_range_dither = scribble.visibility_range_dither;
#endif
    return out;
}

const JITTER_PERIOD: f32 = 1.0;

@vertex
fn vertex(vertex: Vertex, @builtin(vertex_index) vertex_index: u32) -> ScribbleOutput {
    var out: ScribbleOutput;

    var seed: u32 = vertex_index + u32(floor(globals.time / JITTER_PERIOD));
    let jitter_mul_pos = mix(0.975, 1.025, rand_f(&seed));
    let jitter_mul_uv = mix(0.99, 1.01, rand_f(&seed));
    out.no_scribble_pos = jitter_mul_pos * vertex.position.xz;

    var world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);

#ifdef VERTEX_NORMALS
    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        vertex.normal,
        vertex.instance_index
    );
#endif

#ifdef VERTEX_POSITIONS
    out.world_position = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(jitter_mul_pos * vertex.position, 1.0));
    out.position = position_world_to_clip(out.world_position.xyz);
#endif


#ifdef VERTEX_UVS
    let offset_uv = vertex.uv - vec2(0.5);
    out.uv = jitter_mul_uv * jitter_mul_pos * offset_uv + vec2(0.5);
#endif

#ifdef VERTEX_UVS_B
    let offset_uv_b = vertex.uv_b - vec2(0.5);
    out.uv_b = jitter_mul_uv * jitter_mul_pos * offset_uv_b + vec2(0.5);
#endif

#ifdef VERTEX_TANGENTS
    out.world_tangent = mesh_functions::mesh_tangent_local_to_world(
        world_from_local,
        vertex.tangent,
        vertex.instance_index
    );
#endif

#ifdef VERTEX_COLORS
    out.color = vertex.color;
#endif

#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex.instance_index;
#endif

#ifdef VISIBILITY_RANGE_DITHER
    out.visibility_range_dither = mesh_functions::get_visibility_range_dither_level(
        vertex.instance_index, world_from_local[3]);
#endif

    return out;
}

// in mesh local units
const GRID_SPACING: f32 = 0.4;
const PAPER_COLOR: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
const GRID_COLOR: vec3<f32> = vec3<f32>(0.57, 0.79, 0.83);
const GRID_START_DIST: f32 = 0.01;
const GRID_END_DIST: f32 = 0.03;

@fragment
fn fragment(
    scribble_in: ScribbleOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    let no_scribble_pos = scribble_in.no_scribble_pos;
    let in = vertex_from_scribble(scribble_in);
    // generate a PbrInput struct from the StandardMaterial bindings
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    let nearest_grid_corner = floor(no_scribble_pos / GRID_SPACING) * GRID_SPACING;
    let to_corner = nearest_grid_corner - no_scribble_pos;
    var gridness: f32;
    if abs(to_corner.x) < abs(to_corner.y) {
        gridness = smoothstep(GRID_END_DIST, GRID_START_DIST, abs(to_corner.x));
    } else {
        gridness = smoothstep(GRID_END_DIST, GRID_START_DIST, abs(to_corner.y));
    }

    let paper_color = mix(PAPER_COLOR, GRID_COLOR, gridness);
    let alpha = pbr_input.material.base_color.a;
    let sampled_color = pbr_input.material.base_color.rgb;
    pbr_input.material.base_color = vec4(mix(paper_color, sampled_color, alpha), 1.0);

    // alpha discard
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

#ifdef PREPASS_PIPELINE
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    // apply lighting
    out.color = apply_pbr_lighting(pbr_input);

    // apply in-shader post processing (fog, alpha-premultiply, and also tonemapping, debanding if the camera is non-hdr)
    // note this does not include fullscreen postprocessing effects like bloom.
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);

#endif

    return out;
}