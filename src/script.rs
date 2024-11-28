use std::{borrow::Cow, error::Error, fmt::Display, time::Duration};

use bevy::{
    asset::{io::Reader, AssetLoader}, ecs::{component::{ComponentId, StorageType}, world::DeferredWorld}, prelude::*, scene::ron::de, time::Stopwatch
};
use nom::{
    branch::alt,
    bytes::complete::{tag, take_till, take_till1},
    character::complete::{line_ending, multispace0},
    combinator::{eof, peek, rest, verify},
    error::ParseError,
    multi::{fold_many0, many0},
    sequence::{delimited, preceded, terminated, tuple},
    IResult, InputLength, Parser,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answer {
    pub answer: String,
    pub response: String,
    pub is_end: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnswerBlock {
    /// a single answer, so only the one string is provided
    Single(String),
    /// multiple answers, each with a response
    Many(Vec<Answer>),
}

impl AnswerBlock {
    pub fn get_answer(&self, answer: usize) -> Option<&String> {
        match self {
            Self::Single(s) => (answer == 0).then(|| s),
            Self::Many(choices) => choices.get(answer).map(|answer| &answer.answer)
        }
    }
    pub fn get_response(&self, answer: usize) -> Option<&String> {
        match self {
            Self::Single(s) => None,
            Self::Many(choices) => choices.get(answer).map(|answer| &answer.response)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScriptEntry {
    Prompt {
        prompt: String,
        choices: AnswerBlock,
    },
    Line(String),
}

fn line_starter<'a>(starter: &'a str) -> impl FnMut(&'a str) -> IResult<&'a str, &'a str> {
    preceded(multispace0, tag(starter))
}

fn till_newline0(input: &str) -> IResult<&str, &str> {
    take_till(|c| c == '\r' || c == '\n')(input)
}

fn till_newline1(input: &str) -> IResult<&str, &str> {
    take_till1(|c| c == '\r' || c == '\n')(input)
}

fn prompt<'a>(lines: &'a str) -> IResult<&'a str, &'a str> {
    delimited(
        line_starter(">"),
        till_newline1.map(str::trim),
        Parser::or(line_ending, eof),
    )(lines)
}

fn answer<'a>(lines: &'a str) -> IResult<&'a str, &'a str> {
    delimited(
        line_starter("-"),
        till_newline1.map(str::trim),
        Parser::or(line_ending, eof),
    )(lines)
}

fn end<'a>(lines: &'a str) -> IResult<&'a str, &'a str> {
    delimited(
        line_starter("!"),
        till_newline1.map(str::trim),
        Parser::or(line_ending, eof),
    )(lines)
}

fn line<'a>(lines: &'a str) -> IResult<&'a str, &'a str> {
    verify(terminated(till_newline1.map(str::trim), Parser::or(line_ending, eof)), |s: &str| s.len() > 0)(lines)
}

fn empty<'a>(lines: &'a str) -> IResult<&'a str, &'a str> {
    verify(terminated(till_newline0, line_ending), |s: &str| s.trim().len() == 0)(lines)
}

/// skips while the first parser has input. kinda like preceded but runs the first parser until it fails.
/// maybe this exists in nom already but i didn't find something equivalent
fn skip_while<I, O1, O2, F, S, E>(to_skip: S, parser: F) -> impl FnMut(I) -> IResult<I, O1, E>
where
    I: Clone + InputLength,
    F: Parser<I, O1, E>,
    S: Parser<I, O2, E>,
    E: ParseError<I>,
{
    preceded(fold_many0(to_skip, || (), |_, _| ()), parser)
}

fn prompt_block<'a>(lines: &'a str) -> IResult<&'a str, ScriptEntry> {
    let (lines, (prompt, answer_block)) = (tuple((
        skip_while(empty, prompt),
        alt((
            tuple((skip_while(empty, answer), peek(skip_while(empty, prompt))))
                .map(|(answer, _)| AnswerBlock::Single(answer.into())),
            many0(alt((
                tuple((skip_while(empty, answer), skip_while(empty, end))).map(|(answer, line)| {
                    Answer {
                        answer: answer.into(),
                        response: line.into(),
                        is_end: true,
                    }
                }),
                tuple((skip_while(empty, answer), skip_while(empty, line))).map(
                    |(answer, line)| Answer {
                        answer: answer.into(),
                        response: line.into(),
                        is_end: false,
                    },
                ),
            )))
            .map(AnswerBlock::Many),
        )),
    )))(lines)?;
    Ok((
        lines,
        ScriptEntry::Prompt {
            prompt: prompt.into(),
            choices: answer_block,
        },
    ))
}

fn line_block<'a>(lines: &'a str) -> IResult<&'a str, ScriptEntry> {
    skip_while(empty, line)(lines).map(|(input, line)| (input, ScriptEntry::Line(line.into())))
}

fn parse_entries<'a>(lines: &'a str) -> IResult<&'a str, Vec<ScriptEntry>> {
    many0(alt((prompt_block, line_block)))(lines)
}

#[derive(Debug, Asset, TypePath, PartialEq, Eq)]
pub struct Script {
    raw: Cow<'static, str>,
    entries: Vec<ScriptEntry>,
}

impl Script {
    pub fn from_raw(
        raw: impl Into<Cow<'static, str>>,
    ) -> Result<Self, nom::Err<nom::error::Error<String>>> {
        let raw: Cow<'static, str> = raw.into();
        let (extra_input, entries) = parse_entries(&raw).map_err(|e| e.map_input(|s| s.into()))?;
        Ok(Script { raw, entries })
    }

    pub fn get_entry(&self, index: usize) -> Option<&ScriptEntry> {
        self.entries.get(index)
    }
}

#[derive(Default)]
struct ScriptLoader;

#[derive(Debug)]
enum ScriptLoadError {
    Io(std::io::Error),
    Utf8(std::string::FromUtf8Error),
    Parse(nom::Err<nom::error::Error<String>>),
}

impl Display for ScriptLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => e.fmt(f),
            Self::Utf8(e) => e.fmt(f),
            Self::Parse(e) => e.fmt(f),
        }
    }
}

impl Error for ScriptLoadError {
    fn cause(&self) -> Option<&dyn Error> {
        match self {
            Self::Io(e) => Some(e),
            Self::Utf8(e) => Some(e),
            Self::Parse(e) => Some(e),
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
        reader
            .read_to_end(&mut buf)
            .await
            .map_err(ScriptLoadError::Io)?;
        Script::from_raw(String::from_utf8(buf).map_err(ScriptLoadError::Utf8)?)
            .map_err(ScriptLoadError::Parse)
    }
    fn extensions(&self) -> &[&str] {
        &["txt"]
    }
}

/// triggered on a [`ScriptRunner`] to update its state in some way
#[derive(Event)]
pub enum UpdateRunner {
    /// immediately finish displaying the current text
    FinishLine,
    /// immediately progress to the next line (ignored if currently choosing an answer)
    NextLine,
    /// if the index is invalid or the runner isn't currently choosing an answer this is ignored
    ChooseAnswer(usize),
}

/// triggered on a [`ScriptRunner`] to indicate certain state changes
#[derive(Event)]
pub enum RunnerUpdated {
    Finished,
    NoScript
}

// a nicer version of this would have runners/choices automatically add/remove each other with 
// component hooks but that's not insignificant effort and i need to get this done
#[derive(Component)]
#[require(Text)]
#[component(on_add = ScriptRunner::on_add)]
pub struct ScriptRunner {
    script: Handle<Script>,
    choices_display: Entity,
    /// index of the current ScriptEntry
    current_entry: usize,
    /// if this is None, the text currently being displayed is either a single line (if 
    /// `current_entry` refers to a `ScriptEntry::Line`) or a prompt (`current_entry` refers to a 
    /// `ScriptEntry::Prompt`). if `current_entry` refers to a `Prompt` with 
    /// `choices = AnswerBlock::Many(choice_vec)`, this is `Some(i)` where 
    /// `choice_vec[i].response` is the text currently being displayed.
    current_answer: Option<usize>,
    /// number of bytes displayed so far
    displayed_bytes: usize,
    display_timer: Timer,
    /// time since the current line starting displaying in timer ticks
    display_ticks: u32,
    /// the tick time of the last displayed char
    last_displayed_tick: u32,
}

impl ScriptRunner {
    pub fn new(script: Handle<Script>, choices_display: Entity, text_speed: f32) -> Self {
        ScriptRunner {
            script,
            choices_display,
            current_entry: 0,
            current_answer: None,
            displayed_bytes: 0,
            display_timer: Timer::new(Duration::from_secs_f32(1. / text_speed), TimerMode::Repeating),
            display_ticks: 0,
            last_displayed_tick: 0
        }
    }
    pub fn pause(&mut self) {
        self.display_timer.pause();
    }
    pub fn unpause(&mut self) {
        self.display_timer.unpause();
    }
    pub fn paused(&self) -> bool {
        self.display_timer.paused()
    }
    fn reset_display(&mut self) {
        self.display_timer.reset();
        self.displayed_bytes = 0;
        self.display_ticks = 0;
        self.last_displayed_tick = 0;
    }

    fn on_add(mut deferred_world: DeferredWorld, entity: Entity, _: ComponentId) {
        deferred_world
            .commands()
            .entity(entity)
            .observe(ScriptRunner::on_update_runner_trigger);
    }
    fn on_update_runner_trigger(
        trigger: Trigger<UpdateRunner>,
        mut commands: Commands,
        mut runners: Query<(&mut ScriptRunner, &mut Text)>,
        mut choices_containers: Query<&mut ScriptChoices>,
        scripts: Res<Assets<Script>>
    ) {
        let Ok((mut runner, mut text)) = runners.get_mut(trigger.entity()) else {
            return;
        };
        let runner = &mut *runner;
        let mut runner_commands = commands.entity(trigger.entity());
        let Some(script) = scripts.get(runner.script.id()) else {
            runner_commands.trigger(RunnerUpdated::NoScript);
            return;
        };
        let Some(entry) = script.get_entry(runner.current_entry) else {
            runner_commands.trigger(RunnerUpdated::Finished);
            return;
        };
        match trigger.event() {
            UpdateRunner::FinishLine => {
                match (entry, runner.current_answer) {
                    (ScriptEntry::Line(s), None) 
                        | (ScriptEntry::Prompt { prompt: s, ..}, None) => {
                        text.0.clone_from(s);
                    }
                    (ScriptEntry::Prompt { choices: AnswerBlock::Many(answers), .. }, Some(index)) => {
                        let Some(answer) = answers.get(index) else { return; };
                        text.0.clone_from(&answer.response);
                    }
                    _ => unreachable!()
                }
            }
            UpdateRunner::NextLine => {
                match (entry, runner.current_answer) {
                    (ScriptEntry::Line(_), None) => {
                        runner.current_entry += 1;
                        text.0.clear();
                        runner.reset_display();
                    }
                    // FIXME: should this be counted as a choice, since there's only one? i think not
                    (ScriptEntry::Prompt { .. }, None) => {}
                    (ScriptEntry::Prompt { choices: AnswerBlock::Many(_), .. }, Some(_)) => {
                        runner.current_entry += 1;
                        runner.current_answer = None;
                        text.0.clear();
                        runner.reset_display();
                    }
                    _ => unreachable!()
                }
            }
            &UpdateRunner::ChooseAnswer(index) => {
                match (entry, runner.current_answer) {
                    (_, Some(_)) | (ScriptEntry::Line(_), _) => {}
                    (ScriptEntry::Prompt { choices: AnswerBlock::Single(_), .. }, None) => {
                        if index == 0 {
                            runner.current_entry += 1;
                            text.0.clear();
                            runner.reset_display();
                        }
                    }
                    (ScriptEntry::Prompt { choices: AnswerBlock::Many(answers), .. }, None) => {
                        if index >= answers.len() {
                            return;
                        };
                        runner.current_answer = Some(index);
                        text.0.clear();
                        runner.reset_display();
                    }
                }
            }
        }
    }
}

#[derive(Component)]
#[require(Node)]
pub struct ScriptChoices {
    runner: Entity,
    // for highlighting n stuff
    active_choice: usize,
}

impl ScriptChoices {
    pub fn new(runner: Entity) -> Self {
        ScriptChoices {
            runner,
            active_choice: 0,
        }
    }

    fn show_choices_systems() {

    }
}

#[derive(Component)]
#[require(Text)]
pub struct ScriptChoice(usize);

fn update_script_runner_text(
    mut commands: Commands,
    mut runners: Query<(&mut ScriptRunner, &mut Text)>,
    mut choice_displays: Query<(&ScriptChoice, &Node, &Children)>,
    scripts: Res<Assets<Script>>,
    time: Res<Time>
) {
    for (mut runner, mut text) in runners.iter_mut() {
        let runner = &mut *runner;
        runner.display_ticks += runner.display_timer.tick(time.delta()).times_finished_this_tick();

        let Some(script) = scripts.get(runner.script.id()) else {
            continue;
        };
        let Some(entry) = script.get_entry(runner.current_entry) else {
            continue;
        };
        let mut extend_text = |s: &str| {
            let mut new_bytes = 0;
            // oh no
            text.0.extend(
                s[runner.displayed_bytes..]
                    .chars()
                    .map(|c| {
                        match c {
                            '.' | '?' | '-' | '!' => (c, 6),
                            ',' => (c, 3),
                            _ => (c, 1)
                        }
                    })
                    .take_while(|(_, ticks)| {
                        if runner.last_displayed_tick < runner.display_ticks {
                            runner.last_displayed_tick += ticks;
                            true
                        } else {
                            false
                        }
                    })
                    .map(|(c, _)| {                        
                        new_bytes += c.len_utf8();
                        c
                    })
            );
            runner.displayed_bytes += new_bytes;
        };
        let text_len;
        match (entry, runner.current_answer) {
            (ScriptEntry::Line(line), None) => {
                extend_text(line);
                text_len = line.as_bytes().len();
            }
            (ScriptEntry::Prompt { prompt, .. }, None) => {
                extend_text(prompt);
                text_len = prompt.as_bytes().len();
            }
            (ScriptEntry::Prompt { choices: AnswerBlock::Many(answers), .. }, Some(answer_index)) => {
                extend_text(&answers[answer_index].response);
                text_len = answers[answer_index].response.as_bytes().len();
            }
            _ => unreachable!()
        }
        if runner.displayed_bytes >= text_len {
            // finished displaying?
        }
    }
}

pub struct ScriptPlugin;

impl Plugin for ScriptPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<Script>()
            .init_asset_loader::<ScriptLoader>()
            .add_systems(Update, update_script_runner_text);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_script_parses() {
        let empties = &[
            "",
            "   ",
            "\t ",
        ];
        for &empty in empties {
            assert_eq!(
                Script::from_raw(empty), 
                Ok(Script { raw: Cow::Borrowed(empty), entries: vec![] })
            );
        }
    }

    #[test]
    fn single_line_parses() {
        let lines = &[
            "test",
            " test",
            "test ",
            "test\n",
            " test \r\n"
        ];
        for &line in lines {
            assert_eq!(
                Script::from_raw(line), 
                Ok(Script { 
                    raw: Cow::Borrowed(line), 
                    entries: vec![ScriptEntry::Line(line.trim().into())] 
                })
            );
        }
        
    }

    #[test]
    fn prompt_block_parses() {
        let single_answer = ">prompt\r\n\n\n-   answer\n>next prompt\n-next 1\nnext response 1\n-next 2\nnext response 2";
        assert_eq!(
            Script::from_raw(single_answer), 
            Ok(Script { 
                raw: single_answer.into(), 
                entries: vec![
                    ScriptEntry::Prompt {
                        prompt: "prompt".into(),
                        choices: AnswerBlock::Single("answer".into())
                    },
                    ScriptEntry::Prompt { 
                        prompt: "next prompt".into(), 
                        choices: AnswerBlock::Many(vec![
                            Answer { answer: "next 1".into(), response: "next response 1".into(), is_end: false },
                            Answer { answer: "next 2".into(), response: "next response 2".into(), is_end: false },
                        ]) 
                    }
                ]
            })
        );

        let multi_answer = ">prompt\n\n\r\n- answer1  \r\n response1\t\n-answer2\r\nresponse2\n-end answer\n\r\n!end response";
        assert_eq!(
            Script::from_raw(multi_answer), 
            Ok(Script { 
                raw: multi_answer.into(), 
                entries: vec![ScriptEntry::Prompt { 
                    prompt: "prompt".into(), 
                    choices: AnswerBlock::Many(vec![
                        Answer { answer: "answer1".into(), response: "response1".into(), is_end: false },
                        Answer { answer: "answer2".into(), response: "response2".into(), is_end: false },
                        Answer { answer: "end answer".into(), response: "end response".into(), is_end: true },
                    ]) 
                }]
            })
        );
    }

    #[test]
    fn mixed_blocks_parse() {
        let to_parse = "line 1\n> prompt \r\n-   answer\n response  \r\n line 2\n \n line 3 \r\n";
        assert_eq!(
            Script::from_raw(to_parse),
            Ok(Script {
                raw: to_parse.into(),
                entries: vec![
                    ScriptEntry::Line("line 1".into()),
                    ScriptEntry::Prompt { 
                        prompt: "prompt".into(), 
                        choices: AnswerBlock::Many(vec![Answer { 
                            answer: "answer".into(), 
                            response: "response".into(), 
                            is_end: false 
                        }])
                    },
                    ScriptEntry::Line("line 2".into()),
                    ScriptEntry::Line("line 3".into()),
                ]
            })
        );
    }
}