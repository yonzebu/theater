## What is this?
This is a project I made for myself using the [Bevy](https://bevyengine.org/) game engine. I consider it "finished" in the sense that I don't plan to update or change it in any way unless I decide to reuse code or assets for something else later. It does not include all intended assets because I was concerned about publicly releasing certain assets (notably, a video called `show.mp4` is missing, although other videos would probably allow the project to run in its place).

## Why release this?
Because it's a thing I made and I'm happy I made it and if there's even a small chance someone else can learn from or use any of the things I made, I'm glad to provide that chance. The most interesting parts of this are probably the hacked-together custom light source (see `screen_light.rs`) and the video decoding/playback (built off `ffmpeg-next`, see `video.rs`).

## Credits
This would not have been possible without [Bevy](https://bevyengine.org/) (in addition to various other dependencies, I don't want to list them all here, check the Cargo.toml). In addition, this uses [a material](https://polyhaven.com/a/leather_red_03) from [Polyhaven](https://polyhaven.com/) and models made using [Blender](https://www.blender.org/).

## License
Except where noted otherwise (see below), everything in this repository is dual-licensed under either:

- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.

Assets (everything under the `assets` directory except `.wgsl` files) are released under CC0 1.0 Universal ([CC0](CC0) or https://creativecommons.org/publicdomain/zero/1.0/).