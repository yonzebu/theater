use std::{collections::VecDeque, marker::PhantomData, path::{Path, PathBuf}, sync::{mpsc::{channel, Receiver, Sender, TryRecvError}, Mutex}, task::{Context, Waker}};

use bevy::{asset::{io::AssetSourceId, AssetLoader, ErasedAssetLoader, RenderAssetUsages}, prelude::*, render::{render_resource::{Extent3d, TextureDimension, TextureFormat}, texture::ImageLoader}, tasks::{futures_lite::FutureExt, AsyncComputeTaskPool, ComputeTaskPool, Task}};
use ffmpeg_next::{
    self as ffmpeg, codec, decoder, ffi::EAGAIN, format::{context::Input, Pixel}, frame, media, software::scaling::{self, Flags}, Packet, Rational
};

struct ScalingContext(scaling::Context);
// SAFETY: i'm like 90% sure this is fine
unsafe impl Send for ScalingContext {}

#[derive(Debug)]
struct DecodedFrame {
    width: u32,
    height: u32,
    data: Vec<u8>,
    pts: i64
}

impl From<DecodedFrame> for Image {
    fn from(value: DecodedFrame) -> Self {
        Image::new(
            Extent3d { width: value.width, height: value.height, depth_or_array_layers: 1},
            TextureDimension::D2,
            value.data,
            TextureFormat::Rgba8Unorm,
            RenderAssetUsages::all()
        )
    }
}

struct VideoDecoder {
    input: Input,
    stream_index: usize,
    decoder: decoder::Video,
    scaler: ScalingContext,
    latest_pts: Option<i64>,
    finished: bool,
    send_frames: Sender<DecodedFrame>,
    send_self: Sender<VideoDecoder>,
}
// SAFETY: i don't plan on accessing this when i don't have mutable access anyways 
unsafe impl Sync for VideoDecoder {}

impl VideoDecoder {
    fn decode_until_pts(mut self, target_pts: i64) {
        while self.latest_pts.is_none_or(|pts| pts <= target_pts) {
            let mut should_send_packet;
            let mut packet;
            loop {
                packet = Packet::empty();
                match packet.read(&mut self.input) {
                    Ok(()) => should_send_packet = true,
                    Err(ffmpeg::Error::Other { errno: EAGAIN }) => should_send_packet = false,
                    Err(ffmpeg::Error::Eof) => {
                        self.finished = true;
                        self.finish_task();
                        return;
                    },
                    Err(e) => panic!("packet read error: {e:?}")
                }
                if packet.stream() == self.stream_index {
                    break;
                }
            }
            if should_send_packet {
                self.decoder.send_packet(&packet).unwrap();
            }
        
            let mut decoded;
            let mut scaled;
            loop {
                decoded = frame::Video::empty();
                match self.decoder.receive_frame(&mut decoded) {
                    Ok(()) => {
                        self.latest_pts = decoded.pts();
                        scaled = frame::Video::empty();
                        self.scaler.0.run(&decoded, &mut scaled).unwrap();

                        self.send_frames.send(DecodedFrame { 
                            width: scaled.width(), 
                            height: scaled.height(), 
                            data: scaled.data(0).to_owned(),
                            pts: decoded.pts().unwrap(),
                        }).unwrap();
                    }
                    Err(ffmpeg::Error::Other { errno: EAGAIN }) => break,
                    Err(e) => panic!("receive frame error: {e:?}")
                }
            }
        }

        self.finish_task();
    }

    fn finish_task(self) {
        let send_self = self.send_self.clone();
        send_self.send(self).unwrap();
    }
}

#[derive(Debug)]
pub struct BufferedFrame {
    pub image: Handle<Image>,
    pts: i64
}

#[derive(Asset, TypePath)]
pub struct VideoStream {
    decoder: Option<VideoDecoder>,
    recv_decoder: Mutex<Receiver<VideoDecoder>>,
    recv_frames: Mutex<Receiver<DecodedFrame>>,
    /// frames that are currently loaded and available to the renderer
    pub buffered_frames: VecDeque<BufferedFrame>,
    pub playing: bool,
    /// in seconds (i think)
    progress: Rational,
    time_base: Rational,
    start_time: i64,
}

impl VideoStream {
    pub fn new(source: impl AsRef<Path>) -> Result<Self, ffmpeg::Error> {
        let input = ffmpeg::format::input(&source)?;
        let stream = input
            .streams()
            .best(media::Type::Video)
            .ok_or(ffmpeg::Error::StreamNotFound)?;
        println!(
            "stream stuff: time_base = {:?}, start_time={}, duration = {}, frames = {}, rate = {:?}, avg frame rate = {:?}",
            stream.time_base(), stream.start_time(), stream.duration(), stream.frames(), stream.rate(), stream.avg_frame_rate()
        );
        let time_base = stream.time_base();
        let start_time = stream.start_time();
        let stream_index = stream.index();
        let decoder_ctx = codec::Context::from_parameters(stream.parameters())?;
        let decoder = decoder_ctx.decoder().video()?;
        let scaler = ScalingContext(decoder.converter(Pixel::RGBA)?);
        let (send_frames, recv_frames) = channel();
        let (send_decoder, recv_decoder) = channel();
        Ok(VideoStream {
            decoder: Some(VideoDecoder {
                input,
                stream_index,
                decoder,
                scaler,
                latest_pts: None,
                finished: false,
                send_self: send_decoder,
                send_frames
            }),
            recv_decoder: Mutex::new(recv_decoder),
            recv_frames: Mutex::new(recv_frames),
            buffered_frames: VecDeque::new(),
            playing: false,
            progress: Rational::new(start_time as i32, 1) * time_base,
            time_base,
            start_time
        })
    }

    pub fn is_finished(&self) -> Option<bool> {
        self.decoder.as_ref().map(|decoder| decoder.finished)
    }
}

fn update_videos(
    mut videos: ResMut<Assets<VideoStream>>,
    mut images: ResMut<Assets<Image>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
) {
    for (_, video) in videos.iter_mut() {
        let time_base = video.time_base;
        if video.playing {
            let dt: Rational = time.delta_secs_f64().into();
            video.progress = video.progress + dt;
        }
        let lookahead_progress = video.progress + Rational(5, 1);
        if let Some(mut decoder) = video.decoder.take() {
            if video.buffered_frames.back().is_none_or(|frame| Rational(frame.pts as i32, 1) <= lookahead_progress / time_base)
                && !decoder.finished 
            {
                if let Some(frame) = video.buffered_frames.back() {
                    decoder.latest_pts = Some(frame.pts);
                }
                AsyncComputeTaskPool::get().spawn(async move {
                    decoder.decode_until_pts(Into::<f64>::into(lookahead_progress / time_base) as i64)
                }).detach();
            } else {
                // no need to decode more frames yet
                video.decoder = Some(decoder);
            }
        } else {
            if let Ok(decoder) = video.recv_decoder.get_mut().unwrap().try_recv() {
                video.decoder = Some(decoder);
            }
        }
        
        while let Ok(frame) = video.recv_frames.get_mut().unwrap().try_recv() {
            let pts = frame.pts;
            video.buffered_frames.push_back(BufferedFrame { image: images.add(frame), pts });
        }
    
        let current_pts: f64 = (video.progress / time_base).into();
        let current_pts = current_pts as i64;
        let current = video.buffered_frames.partition_point(|frame| frame.pts < current_pts);
        video.buffered_frames.drain(0..current);
    }
}

struct VideoStreamLoader {
    root: PathBuf,
}

impl FromWorld for VideoStreamLoader {
    fn from_world(_world: &mut World) -> Self {
        let root = if let Ok(manifest_dir) = std::env::var("BEVY_ASSET_ROOT") {
            PathBuf::from(manifest_dir)
        } else if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
            PathBuf::from(manifest_dir)
        } else {
            std::env::current_exe()
                .map(|path| path.parent().map(ToOwned::to_owned).unwrap())
                .unwrap()
        }.join("assets");
        VideoStreamLoader {
            root
        }
    }
}

impl AssetLoader for VideoStreamLoader {
    type Asset = VideoStream;
    type Error = ffmpeg::Error;
    type Settings = ();
    async fn load(
        &self,
        reader: &mut dyn bevy::asset::io::Reader,
        _settings: &Self::Settings,
        load_context: &mut bevy::asset::LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        VideoStream::new(self.root.join(load_context.path()))
    }
    fn extensions(&self) -> &[&str] {
        &["mp4"]
    }
}

pub struct VideoPlugin;

impl Plugin for VideoPlugin {
    fn build(&self, app: &mut App) {
        ffmpeg::init().unwrap();
        app
            .init_asset::<VideoStream>()
            .init_asset_loader::<VideoStreamLoader>()
            .add_systems(PostUpdate, update_videos);
    }
}
