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
    utils::rand_f,
    mesh_bindings::mesh,
    mesh_functions,
    mesh_view_bindings::{globals, point_shadow_textures_comparison_sampler},
    mesh_view_bindings as view_bindings,
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    view_transformations::position_world_to_clip
}

fn sample_shadow_map_hardware(
    depth_texture: texture_depth_2d, 
    light_local: vec2<f32>, 
    depth: f32, 
    array_index: i32
) -> f32 {
    return textureSampleCompare(
        depth_texture,
        view_bindings::directional_shadow_textures_comparison_sampler,
        light_local,
        depth,
    );
}

// these are taken from bevy's default biases for spot lights
const DEPTH_BIAS: f32 = 0.02;
// ...except bevy also multiplies the normal bias by some values, so these are precalculated based on approximate expected shadow map size (1920x1080)
// ...and then futzed with using a fudge factor a little bit
const NORMAL_BIAS: f32 = 1.8 * 1.4142135 * 0.002 * 2.0;

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

    // not sure if i should use soft shadows or not
    // return sample_shadow_map_pcss(
    //     shadow_uv, depth, array_index, SPOT_SHADOW_TEXEL_SIZE, (*light).soft_shadow_size);

    // return sample_shadow_map(shadow_uv, depth, array_index, SPOT_SHADOW_TEXEL_SIZE);

    return textureSampleCompare(shadow_map, shadow_sampler, shadow_uv, ndc_pos.z);
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