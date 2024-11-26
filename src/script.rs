use std::{borrow::Cow, error::Error, fmt::Display};

use bevy::{asset::{AssetLoader, io::Reader}, prelude::*};
use nom::{branch::alt, bytes::complete::{tag, take_while}, character::complete::{line_ending, multispace0}, combinator::{eof, peek, rest}, error::ParseError, multi::{fold_many0, many0, many0_count}, sequence::{delimited, preceded, terminated, tuple}, IResult, InputLength, InputTakeAtPosition, Parser};

#[derive(Debug, Clone)]
struct Answer {
    answer: String,
    response: String,
    is_end: bool,
}

#[derive(Debug, Clone)]
enum AnswerBlock {
    Single(String),
    Multi(Vec<Answer>)
}

#[derive(Debug, Clone)]
enum ScriptEntry {
    Prompt {
        prompt: String,
        choices: AnswerBlock,
    },
    Line(String),
}

fn line_starter<'a>(starter: &'a str) -> impl FnMut(&'a str) -> IResult<&'a str, &'a str> {
    preceded(multispace0, tag(starter))
}

fn strip_whitespace<'a>(input: &'a str) -> IResult<&'a str, &'a str> {
    delimited(multispace0, rest, multispace0)(input)
}

fn prompt<'a>(lines: &'a str) -> IResult<&'a str, &'a str> {
    delimited(line_starter(">"), strip_whitespace, Parser::or(line_ending, eof))(lines)
}

fn answer<'a>(lines: &'a str) -> IResult<&'a str, &'a str> {
    delimited(line_starter("-"), strip_whitespace, Parser::or(line_ending, eof))(lines)
}

fn end<'a>(lines: &'a str) -> IResult<&'a str, &'a str> {
    delimited(line_starter("!"), strip_whitespace, Parser::or(line_ending, eof))(lines)
}

fn line<'a>(lines: &'a str) -> IResult<&'a str, &'a str> {
    terminated(strip_whitespace, Parser::or(line_ending, eof))(lines)
}

fn empty<'a>(lines: &'a str) -> IResult<&'a str, &'a str> {
    terminated(strip_whitespace, Parser::or(line_ending, eof))(lines)
}

/// skips while the first parser has input. kinda like preceded but runs the first parser until it fails.
/// maybe this exists in nom already but i didn't find something equivalent
fn skip_while<I, O1, O2, F, S, E>(to_skip: S, parser: F) -> impl FnMut(I) -> IResult<I, O1, E>
where
    I: Clone + InputLength,
    F: Parser<I, O1, E>,
    S: Parser<I, O2, E>,
    E: ParseError<I>
{
    preceded(fold_many0(to_skip, || (), |_, _| ()), parser)
}

fn prompt_block<'a>(lines: &'a str) -> IResult<&'a str, ScriptEntry> {
    let (lines, (prompt, answer_block)) = (tuple((
        skip_while(empty, prompt), 
        alt((
            tuple((
                skip_while(empty, answer), 
                peek(skip_while(empty, prompt))
            ))
                .map(|(answer, _)| AnswerBlock::Single(answer.into())),
            many0(
                alt((
                    tuple((
                        skip_while(empty, answer),
                        skip_while(empty, end)
                    ))
                        .map(|(answer, line)| Answer { answer: answer.into(), response: line.into(), is_end: true }),
                    tuple((
                        skip_while(empty, answer),
                        skip_while(empty, line)
                    ))
                        .map(|(answer, line)| Answer { answer: answer.into(), response: line.into(), is_end: false }),
                ))
            )
                .map(AnswerBlock::Multi)
        ))
    )))(lines)?;
    Ok((lines, ScriptEntry::Prompt {
        prompt: prompt.into(),
        choices: answer_block
    }))
}

fn line_block<'a>(lines: &'a str) -> IResult<&'a str, ScriptEntry> {
    skip_while(empty, line)(lines).map(|(input, line)| (input, ScriptEntry::Line(line.into())))
}

fn parse_entries<'a>(lines: &'a str) -> IResult<&'a str, Vec<ScriptEntry>> {
    many0(alt((
        prompt_block,
        line_block
    )))(lines)
}

#[derive(Debug, Asset, TypePath)]
pub struct Script {
    raw: Cow<'static, str>,
    entries: Vec<ScriptEntry>
}

impl Script {
    pub fn from_raw(raw: impl Into<Cow<'static, str>>) -> Result<Self, nom::Err<nom::error::Error<String>>> {
        let raw: Cow<'static, str> = raw.into();
        let (extra_input, entries) = parse_entries(&raw).map_err(|e| e.map_input(|s| s.into()))?;
        Ok(Script {
            raw,
            entries
        })
    }
}

#[derive(Default)]
struct ScriptLoader;

#[derive(Debug)]
enum ScriptLoadError {
    Io(std::io::Error),
    Utf8(std::string::FromUtf8Error),
    Parse(nom::Err<nom::error::Error<String>>)
}

impl Display for ScriptLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => e.fmt(f),
            Self::Utf8(e) => e.fmt(f),
            Self::Parse(e) => e.fmt(f)
        }
    }
}

impl Error for ScriptLoadError {
    fn cause(&self) -> Option<&dyn Error> {
        match self {
            Self::Io(e) => Some(e),
            Self::Utf8(e) => Some(e),
            Self::Parse(e) => Some(e)
        }
    }
}

impl AssetLoader for ScriptLoader {
    type Asset = Script;
    type Error = ScriptLoadError;
    type Settings = ();
    async fn load(
        &self,
        reader: &mut dyn Reader,
        settings: &Self::Settings,
        load_context: &mut bevy::asset::LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).await.map_err(ScriptLoadError::Io)?;
        Script::from_raw(String::from_utf8(buf).map_err(ScriptLoadError::Utf8)?)
            .map_err(ScriptLoadError::Parse)
    }
    fn extensions(&self) -> &[&str] {
        &["txt"]
    }
}

pub struct ScriptPlugin;

impl Plugin for ScriptPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<Script>()
            .init_asset_loader::<ScriptLoader>();
    }
}