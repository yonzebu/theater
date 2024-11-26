use std::borrow::Cow;

use bevy::prelude::*;
use nom::{branch::alt, bytes::complete::{tag, take_while}, character::complete::{line_ending, multispace0}, combinator::{eof, peek, rest}, multi::{fold_many0, many0, many0_count}, sequence::{delimited, preceded, terminated, tuple}, IResult, InputTakeAtPosition, Parser};

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
    End(String)
}

#[derive(Clone, Copy, Debug)]
// all line &str do not include syntax elements like > or -
enum ScriptLine<'a> {
    Prompt(&'a str),
    // user choice options
    Answer(&'a str),
    // plain lines
    Line(&'a str),
    End(&'a str),
    Empty
}

type NumberedLine<'a> = (usize, ScriptLine<'a>);

#[derive(Debug)]
enum ErrorKind {
    Eof,
    NotPrompt,
    NotAnswer,
    NotEnd,
    NotLine,
    NotEmpty,
    Nom(nom::error::ErrorKind)
}

#[derive(Debug)]
struct ParseError<I>(I, ErrorKind);

type ScriptResult<I, O> = IResult<I, O, ParseError<I>>;

impl<I> nom::error::ParseError<I> for ParseError<I> {
    fn from_error_kind(input: I, kind: nom::error::ErrorKind) -> Self {
        ParseError(input, ErrorKind::Nom(kind))
    }
    fn append(input: I, kind: nom::error::ErrorKind, other: Self) -> Self {
        other
    }
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

fn parse_line<'a>(lines: &'a str) -> IResult<&'a str, ScriptLine<'a>> {
    alt((
        empty.map(|_| ScriptLine::Empty),
        prompt.map(ScriptLine::Prompt),
        answer.map(ScriptLine::Answer),
        end.map(ScriptLine::End),
        line.map(ScriptLine::Line)
    ))(lines)
}

fn prompt_line<'a, 'b>(lines: &'b [NumberedLine<'a>]) -> ScriptResult<&'b [NumberedLine<'a>], NumberedLine<'a>> {
    match lines.get(0) {
        None => Err(nom::Err::Error(ParseError(lines, ErrorKind::Eof))),
        Some(&numbered_line) if matches!(numbered_line.1, ScriptLine::Prompt(_)) => Ok((&lines[1..], numbered_line)),
        _ => Err(nom::Err::Error(ParseError(lines, ErrorKind::NotPrompt)))
    }
}

fn answer_line<'a, 'b>(lines: &'b [NumberedLine<'a>]) -> ScriptResult<&'b [NumberedLine<'a>], NumberedLine<'a>> {
    match lines.get(0) {
        None => Err(nom::Err::Error(ParseError(lines, ErrorKind::Eof))),
        Some(&numbered_line) if matches!(numbered_line.1, ScriptLine::Answer(_)) => Ok((&lines[1..], numbered_line)),
        _ => Err(nom::Err::Error(ParseError(lines, ErrorKind::NotAnswer)))
    }
}

fn end_line<'a, 'b>(lines: &'b [NumberedLine<'a>]) -> ScriptResult<&'b [NumberedLine<'a>], NumberedLine<'a>> {
    match lines.get(0) {
        None => Err(nom::Err::Error(ParseError(lines, ErrorKind::Eof))),
        Some(&numbered_line) if matches!(numbered_line.1, ScriptLine::End(_)) => Ok((&lines[1..], numbered_line)),
        _ => Err(nom::Err::Error(ParseError(lines, ErrorKind::NotEnd)))
    }
}

fn line_line<'a, 'b>(lines: &'b [NumberedLine<'a>]) -> ScriptResult<&'b [NumberedLine<'a>], NumberedLine<'a>> {
    match lines.get(0) {
        None => Err(nom::Err::Error(ParseError(lines, ErrorKind::Eof))),
        Some(&numbered_line) if matches!(numbered_line.1, ScriptLine::Line(_)) => Ok((&lines[1..], numbered_line)),
        _ => Err(nom::Err::Error(ParseError(lines, ErrorKind::NotLine)))
    }
}

fn empty_line<'a, 'b>(lines: &'b [NumberedLine<'a>]) -> ScriptResult<&'b [NumberedLine<'a>], NumberedLine<'a>> {
    match lines.get(0) {
        None => Err(nom::Err::Error(ParseError(lines, ErrorKind::Eof))),
        Some(&numbered_line) if matches!(numbered_line.1, ScriptLine::Empty) => Ok((&lines[1..], numbered_line)),
        _ => Err(nom::Err::Error(ParseError(lines, ErrorKind::NotEmpty)))
    }
}

fn matches_empty(line: &NumberedLine<'_>) -> bool {
    matches!(line, (_, ScriptLine::Empty))
}

fn skip_while<I, O, F, E, P>(cond: P, parser: F) -> impl FnMut(I) -> IResult<I, O, E> 
where
    I: InputTakeAtPosition,
    E: nom::error::ParseError<I>,
    F: Parser<I, O, E>,
    P: Fn(I::Item) -> bool
{
    preceded(take_while(cond), parser)
}

fn prompt_block<'a, 'b>(lines: &'b [NumberedLine<'a>]) -> ScriptResult<&'b [NumberedLine<'a>], ScriptNode> {
    let (a, b) = (tuple((
        skip_while(matches_empty, prompt_line), 
        alt((
            tuple((
                skip_while(matches_empty, answer_line), 
                peek(skip_while(matches_empty, prompt_line))
            ))
                .map(|((answer, _), (prompt, _))| vec![Answer { line: answer, next: prompt }]),
            many0(
                tuple((
                    skip_while(matches_empty, answer_line),
                    skip_while(matches_empty, line_line)
                ))
                    .map(|((answer, _), (response, _))| Answer { line: answer, next: response })
            )
        ))
    )))(lines)?;
}

fn parse_lines<'a>(lines: &'a str) -> IResult<&'a str, Vec<ScriptNode>> {
    let mut index = 0_usize;
    let (input, nodes) = fold_many0(
        parse_line.map(move |line| {
            let i = index;
            index += 1;
            (i, line)
        }), 
        Vec::new, 
        |mut nodes, (index, line)| {

        }
    )(lines)?;
}

#[derive(Debug, Asset, TypePath)]
pub struct Script {
    raw: Cow<'static, str>,
    lines: Vec<String>,
    nodes: Vec<ScriptNode>
}

impl Script {
    pub fn from_raw(raw: impl Into<Cow<'static, str>>) -> Option<Self> {
        let raw: Cow<'static, str> = raw.into();
        let mut lines = Vec::new();
        for line in raw.lines().filter(|s| !s.is_empty()) {
            lines.push(line.into());
        }
        None
    }
}

struct ScriptLoader;

pub struct ScriptPlugin;

impl Plugin for ScriptPlugin {
    fn build(&self, app: &mut App) {

    }
}