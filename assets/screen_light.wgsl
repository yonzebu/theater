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
    mesh_view_bindings::{globals, point_shadow_textures_comparison_sampler},
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

struct ScreenLightUniform {
    image_size: vec4<u32>,
    clip_from_world: mat4x4<f32>,
}

#ifndef PREPASS_FRAGMENT
@group(2) @binding(96) var<uniform> screen_light: ScreenLightUniform;
#endif
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
    let clip_pos = screen_light.clip_from_world * in.world_position;
    let ndc_pos = clip_pos.xyz / clip_pos.w;
    let shadow_uv = ndc_pos.xy * vec2(0.5, -0.5) + vec2(0.5);
#ifndef PREPASS_FRAGMENT
    out.color = vec4(vec3(textureSample(shadow_map, screen_sampler, shadow_uv)), 1.0);
#else
    out.color.a = 1.0;
#endif
    return out;
}