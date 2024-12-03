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
#endif // PREPASS_PIPELINE
#import bevy_pbr::{
    utils::{rand_f, interleaved_gradient_noise},
    utils,
    mesh_bindings::mesh,
    mesh_functions,
    mesh_view_bindings::{globals, point_shadow_textures_comparison_sampler},
    mesh_view_bindings as view_bindings,
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    view_transformations::position_world_to_clip
}
#import bevy_render::maths::PI

fn sample_shadow_map_hardware(
    depth_texture: texture_depth_2d, 
    shadow_sampler: sampler_comparison,
    light_local: vec2<f32>, 
    depth: f32
) -> f32 {
    return textureSampleCompare(
        depth_texture,
        shadow_sampler,
        light_local,
        depth,
    );
}

fn sample_shadow_map_castano_thirteen(
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    light_local: vec2<f32>, 
    depth: f32
) -> f32 {
    let shadow_map_size = vec2<f32>(textureDimensions(shadow_map));
    let inv_shadow_map_size = 1.0 / shadow_map_size;

    let uv = light_local * shadow_map_size;
    var base_uv = floor(uv + 0.5);
    let s = (uv.x + 0.5 - base_uv.x);
    let t = (uv.y + 0.5 - base_uv.y);
    base_uv -= 0.5;
    base_uv *= inv_shadow_map_size;

    let uw0 = (4.0 - 3.0 * s);
    let uw1 = 7.0;
    let uw2 = (1.0 + 3.0 * s);

    let u0 = (3.0 - 2.0 * s) / uw0 - 2.0;
    let u1 = (3.0 + s) / uw1;
    let u2 = s / uw2 + 2.0;

    let vw0 = (4.0 - 3.0 * t);
    let vw1 = 7.0;
    let vw2 = (1.0 + 3.0 * t);

    let v0 = (3.0 - 2.0 * t) / vw0 - 2.0;
    let v1 = (3.0 + t) / vw1;
    let v2 = t / vw2 + 2.0;

    var sum = 0.0;

    sum += uw0 * vw0 * sample_shadow_map_hardware(shadow_map, shadow_sampler, base_uv + (vec2(u0, v0) * inv_shadow_map_size), depth);
    sum += uw1 * vw0 * sample_shadow_map_hardware(shadow_map, shadow_sampler, base_uv + (vec2(u1, v0) * inv_shadow_map_size), depth);
    sum += uw2 * vw0 * sample_shadow_map_hardware(shadow_map, shadow_sampler, base_uv + (vec2(u2, v0) * inv_shadow_map_size), depth);

    sum += uw0 * vw1 * sample_shadow_map_hardware(shadow_map, shadow_sampler, base_uv + (vec2(u0, v1) * inv_shadow_map_size), depth);
    sum += uw1 * vw1 * sample_shadow_map_hardware(shadow_map, shadow_sampler, base_uv + (vec2(u1, v1) * inv_shadow_map_size), depth);
    sum += uw2 * vw1 * sample_shadow_map_hardware(shadow_map, shadow_sampler, base_uv + (vec2(u2, v1) * inv_shadow_map_size), depth);

    sum += uw0 * vw2 * sample_shadow_map_hardware(shadow_map, shadow_sampler, base_uv + (vec2(u0, v2) * inv_shadow_map_size), depth);
    sum += uw1 * vw2 * sample_shadow_map_hardware(shadow_map, shadow_sampler, base_uv + (vec2(u1, v2) * inv_shadow_map_size), depth);
    sum += uw2 * vw2 * sample_shadow_map_hardware(shadow_map, shadow_sampler, base_uv + (vec2(u2, v2) * inv_shadow_map_size), depth);

    return sum * (1.0 / 144.0);
}

fn map(min1: f32, max1: f32, min2: f32, max2: f32, value: f32) -> f32 {
    return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}

fn random_rotation_matrix(scale: vec2<f32>, temporal: bool) -> mat2x2<f32> {
    let random_angle = 2.0 * PI * interleaved_gradient_noise(
        scale, select(1u, view_bindings::globals.frame_count, temporal));
    let m = vec2(sin(random_angle), cos(random_angle));
    return mat2x2(
        m.y, -m.x,
        m.x, m.y
    );
}

fn calculate_uv_offset_scale_jimenez_fourteen(texel_size: f32, blur_size: f32) -> vec2<f32> {
    let shadow_map_size = vec2<f32>(textureDimensions(view_bindings::directional_shadow_textures));

    // Empirically chosen fudge factor to make PCF look better across different CSM cascades
    let f = map(0.00390625, 0.022949219, 0.015, 0.035, texel_size);
    return f * blur_size / (texel_size * shadow_map_size);
}

fn sample_shadow_map_jimenez_fourteen(
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    light_local: vec2<f32>,
    depth: f32,
    texel_size: f32,
    blur_size: f32,
    temporal: bool,
) -> f32 {
    let shadow_map_size = vec2<f32>(textureDimensions(shadow_map));
    let rotation_matrix = random_rotation_matrix(light_local * shadow_map_size, temporal);
    let uv_offset_scale = calculate_uv_offset_scale_jimenez_fourteen(texel_size, blur_size);

    // https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare (slides 120-135)
    let sample_offset0 = (rotation_matrix * utils::SPIRAL_OFFSET_0_) * uv_offset_scale;
    let sample_offset1 = (rotation_matrix * utils::SPIRAL_OFFSET_1_) * uv_offset_scale;
    let sample_offset2 = (rotation_matrix * utils::SPIRAL_OFFSET_2_) * uv_offset_scale;
    let sample_offset3 = (rotation_matrix * utils::SPIRAL_OFFSET_3_) * uv_offset_scale;
    let sample_offset4 = (rotation_matrix * utils::SPIRAL_OFFSET_4_) * uv_offset_scale;
    let sample_offset5 = (rotation_matrix * utils::SPIRAL_OFFSET_5_) * uv_offset_scale;
    let sample_offset6 = (rotation_matrix * utils::SPIRAL_OFFSET_6_) * uv_offset_scale;
    let sample_offset7 = (rotation_matrix * utils::SPIRAL_OFFSET_7_) * uv_offset_scale;

    var sum = 0.0;
    sum += sample_shadow_map_hardware(shadow_map, shadow_sampler, light_local + sample_offset0, depth);
    sum += sample_shadow_map_hardware(shadow_map, shadow_sampler, light_local + sample_offset1, depth);
    sum += sample_shadow_map_hardware(shadow_map, shadow_sampler, light_local + sample_offset2, depth);
    sum += sample_shadow_map_hardware(shadow_map, shadow_sampler, light_local + sample_offset3, depth);
    sum += sample_shadow_map_hardware(shadow_map, shadow_sampler, light_local + sample_offset4, depth);
    sum += sample_shadow_map_hardware(shadow_map, shadow_sampler, light_local + sample_offset5, depth);
    sum += sample_shadow_map_hardware(shadow_map, shadow_sampler, light_local + sample_offset6, depth);
    sum += sample_shadow_map_hardware(shadow_map, shadow_sampler, light_local + sample_offset7, depth);
    return sum / 8.0;
}

fn sample_shadow_map(
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    light_uv: vec2<f32>, 
    depth: f32, 
    texel_size: f32
) -> f32 {
    return sample_shadow_map_castano_thirteen(shadow_map, shadow_sampler, light_uv, depth);
//     return sample_shadow_map_jimenez_fourteen(shadow_map, shadow_sampler, light_uv, depth, texel_size, 1.0, false);
//     return sample_shadow_map_hardware(shadow_map, shadow_sampler, light_uv, depth);
}

const APPROX_TEXEL_SIZE: f32 = 0.002;
// these are taken from bevy's default biases for spot lights
const DEPTH_BIAS: f32 = 0.02;
// ...except bevy also multiplies the normal bias by some values, so these are precalculated based on approximate expected shadow map size (1920x1080)
// ...and then futzed with using a fudge factor a little bit
const NORMAL_BIAS: f32 = 1.8 * 1.4142135 * APPROX_TEXEL_SIZE * 4.0;

fn fetch_screen_light_shadow(
    light_pos: vec3<f32>, 
    light_dir: vec3<f32>,
    world_position: vec3<f32>, 
    surface_normal: vec3<f32>,
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
) -> f32 {
    let light_to_surface = world_position - light_pos;
    let distance_to_light = dot(light_dir, light_to_surface);

    let offset_position =
        world_position
        + (DEPTH_BIAS * normalize(light_to_surface))
        + (surface_normal * NORMAL_BIAS) * distance_to_light;
    let clip_pos = screen_light.clip_from_world * vec4(offset_position, 1.0);
    let ndc_pos = clip_pos.xyz / clip_pos.w;
    let shadow_uv = ndc_pos.xy * vec2(0.5, -0.5) + vec2(0.5);

    return sample_shadow_map(shadow_map, shadow_sampler, shadow_uv, ndc_pos.z, APPROX_TEXEL_SIZE);
}

struct ScreenLightUniform {
    clip_from_world: mat4x4<f32>,
    image_size: vec2<u32>,
    forward_dir: vec3<f32>,
    light_pos: vec3<f32>,
}

@group(2) @binding(96) var<uniform> screen_light: ScreenLightUniform;
@group(2) @binding(97) var shadow_map: texture_depth_2d;
@group(2) @binding(98) var screen_image: texture_2d<f32>;
@group(2) @binding(99) var screen_sampler: sampler;

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {

//     // generate a PbrInput struct from the StandardMaterial bindings
//     var pbr_input = pbr_input_from_standard_material(in, is_front);

//     pbr_input.material.base_color.a = 1.0;

//     // alpha discard
//     pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

// #ifdef PREPASS_PIPELINE
//     let out = deferred_output(in, pbr_input);
// #else
//     var out: FragmentOutput;
//     // apply lighting
//     out.color = apply_pbr_lighting(pbr_input);

//     // TODO: add screen light lighting

//     // apply in-shader post processing (fog, alpha-premultiply, and also tonemapping, debanding if the camera is non-hdr)
//     // note this does not include fullscreen postprocessing effects like bloom.
//     out.color = main_pass_post_lighting_processing(pbr_input, out.color);

// #endif



    var out: FragmentOutput;
    let shadow = fetch_screen_light_shadow(
        screen_light.light_pos, 
        screen_light.forward_dir,
        in.world_position.xyz,
        in.world_normal,
        shadow_map,
        view_bindings::directional_shadow_textures_comparison_sampler
    );
    out.color = vec4(vec3(shadow), 1.0);
    return out;
}