//! Merges the tools of several upstreams under "<server>__<tool>" names.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use rmcp::{
    ErrorData, RoleServer, ServerHandler,
    model::{
        CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, DiscoverResult,
        Implementation, InitializeRequestParams, InitializeResult, ListToolsResult,
        PaginatedRequestParams, ServerCapabilities, ServerConfig, Tool,
    },
    service::RequestContext,
};
use serde_json::{Value, json};

use crate::stats::Stats;
use crate::upstream::{Upstream, UpstreamError};

pub const SEP: &str = "__";

/// Request header naming the servers, separated by commas, that this client is not to get.
/// Sent by `mcp-hub connect` and `mcp-hub headers` for servers with `skipWhenEnv`.
pub const SKIP_HEADER: &str = "x-mcp-hub-skip";

/// Request header naming the agent behind `mcp-hub connect`, which has no `User-Agent`
/// of its own to go by.
pub const CLIENT_HEADER: &str = "x-mcp-hub-client";

/// Request header naming the agent session behind `mcp-hub connect`, which gets its own copy
/// of the `perAgent` servers. See [`crate::connect::session_id`].
pub const SESSION_HEADER: &str = "x-mcp-hub-session";

/// Request header with the folder the agent works in, percent-encoded: the working directory
/// of `mcp-hub connect`, which the agent starts there. The hub gives it to the servers with a
/// `folderArgument` in their calls, so that agents need not.
pub const FOLDER_HEADER: &str = "x-mcp-hub-folder";

/// `path` as a header value: every byte but letters, digits and `/-._~` as `%XX`.
pub fn encode_folder(path: &[u8]) -> String {
    let mut out = String::with_capacity(path.len());
    for &b in path {
        if b.is_ascii_alphanumeric() || b"/-._~".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// The folder an agent names in its request's headers: an absolute path, as UTF-8.
pub fn folder(headers: &hyper::HeaderMap) -> Option<String> {
    let value = headers.get(FOLDER_HEADER)?.as_bytes();
    let mut bytes = Vec::with_capacity(value.len());
    let mut at = 0;
    while at < value.len() {
        if value[at] == b'%' {
            let hex = std::str::from_utf8(value.get(at + 1..at + 3)?).ok()?;
            bytes.push(u8::from_str_radix(hex, 16).ok()?);
            at += 3;
        } else {
            bytes.push(value[at]);
            at += 1;
        }
    }
    let path = String::from_utf8(bytes).ok()?;
    let usable = path.starts_with('/') && path.len() <= 4096 && !path.contains('\0');
    usable.then_some(path)
}

/// The session an agent names in its request's headers, if it is one `mcp-hub connect` makes.
pub fn session(headers: &hyper::HeaderMap) -> Option<String> {
    let value = headers.get(SESSION_HEADER)?.to_str().ok()?;
    let usable = (1..=80).contains(&value.len())
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'-');
    usable.then(|| value.to_owned())
}

/// What a request says about the client that sent it.
#[derive(Default)]
struct Client {
    /// Names of the servers left out for it.
    skip: Vec<String>,
    /// What the agent calls itself, as "claude-code/2.1.0".
    name: Option<String>,
    /// Its session, for the servers that give each one a copy of its own.
    session: Option<String>,
    /// The folder it works in, for the servers with a `folderArgument`.
    folder: Option<String>,
}

impl Client {
    fn of(context: &RequestContext<RoleServer>) -> Self {
        // rmcp hands over the HTTP request each MCP request came in.
        let Some(http) = context.extensions.get::<hyper::http::request::Parts>() else {
            return Self::default();
        };
        let header = |name| http.headers.get(name).and_then(|value| value.to_str().ok());
        let name = header(CLIENT_HEADER).or_else(|| header("user-agent"));
        Self {
            skip: skipped(&http.headers),
            name: name.map(|name| name.chars().take(80).collect()),
            session: session(&http.headers),
            folder: folder(&http.headers),
        }
    }

    /// The folder to give `upstream`'s calls, under the argument it takes it in: only for a
    /// server with a `folderArgument`, and an agent that said where it works.
    fn folder_for<'a>(&'a self, upstream: &'a Upstream) -> Option<(&'a str, &'a str)> {
        let argument = upstream.server.folder_argument.as_deref()?;
        Some((argument, self.folder.as_deref()?))
    }

    /// The servers left out, for an event; nothing when there are none.
    fn skipped(&self) -> Option<&[String]> {
        (!self.skip.is_empty()).then_some(&self.skip[..])
    }

    fn skips(&self, upstream: &Upstream) -> bool {
        self.skip.iter().any(|name| name == upstream.name())
    }
}

/// The servers an agent asks to leave out, from its request's headers.
pub fn skipped(headers: &hyper::HeaderMap) -> Vec<String> {
    let value = headers.get(SKIP_HEADER).and_then(|v| v.to_str().ok());
    let names = value.unwrap_or_default().split(',');
    let names = names.map(|name| name.trim().to_owned());
    names.filter(|name| !name.is_empty()).collect()
}

/// What Claude, Codex and Cursor accept as a tool name.
fn usable_tool_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// The hub's own tool: the descriptions it cut short, in full. Listed while there are any.
pub const DESCRIBE: &str = "hub__describe";

/// In a tools/list result's `_meta`: the version of the list (see [`Hub::tools_version`]).
pub const TOOLS_META: &str = "dev.mcp-hub/tools";

/// `text` cut to about `max` characters, where a paragraph or a line ends if one does
/// nearby, with a note saying where the rest is. `None` if it is short enough as it is.
fn cut(text: &str, max: usize, tool: &str) -> Option<String> {
    // The note takes room too: something only just too long is better left whole.
    let (end, _) = text.char_indices().nth(max + max / 10)?;
    let (end, _) = text[..end].char_indices().nth(max)?;
    let head = &text[..end];
    let at = ["\n\n", "\n", " "].iter().find_map(|sep| {
        let at = head.rfind(sep)?;
        (at >= end / 2).then_some(at)
    });
    let head = head[..at.unwrap_or(end)].trim_end();
    let rest = text[head.len()..].chars().count();
    Some(format!(
        "{head}\n\n[mcp-hub cut the rest, {rest} more characters. To read all of it, \
         call {DESCRIBE} with {{\"tool\": \"{tool}\"}}.]"
    ))
}

/// Cut every description in a tool's input schema, noting in `full` what each was, under
/// the parameter it belongs to.
fn cut_schema(
    schema: &mut Value,
    path: &mut Vec<String>,
    max: usize,
    tool: &str,
    full: &mut Vec<(String, String)>,
) {
    match schema {
        Value::Object(members) => {
            for (key, value) in members.iter_mut() {
                if key == "description"
                    && let Value::String(text) = value
                {
                    if let Some(short) = cut(text, max, tool) {
                        full.push((path.join("."), std::mem::replace(text, short)));
                    }
                    continue;
                }
                // The words of JSON Schema itself say nothing about where this is.
                let named = !matches!(key.as_str(), "properties" | "items" | "$defs");
                if named {
                    path.push(key.clone());
                }
                cut_schema(value, path, max, tool, full);
                if named {
                    path.pop();
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                cut_schema(item, path, max, tool, full);
            }
        }
        _ => {}
    }
}

/// An upstream's tools as the hub lists them.
pub(crate) struct Listed {
    pub tools: Vec<Tool>,
    /// By hub name, the descriptions that were cut, in full: the tool's own under "",
    /// a parameter's under its name.
    pub full: HashMap<String, Vec<(String, String)>>,
}

/// An upstream's tools under their hub names, without those whose name agents would reject,
/// and with descriptions cut to `max_description` characters.
pub(crate) fn hub_tools(server: &str, tools: &[Tool], max_description: Option<usize>) -> Listed {
    let mut listed = Listed {
        tools: Vec::with_capacity(tools.len()),
        full: HashMap::new(),
    };
    for tool in tools {
        let name = format!("{server}{SEP}{}", tool.name);
        if !usable_tool_name(&name) {
            tracing::warn!(
                "[{server}] skipping tool with unusable name: {:?}",
                tool.name
            );
            continue;
        }
        let mut tool = tool.clone();
        if let Some(max) = max_description {
            let mut full = Vec::new();
            if let Some(text) = &mut tool.description
                && let Some(short) = cut(text, max, &name)
            {
                full.push((String::new(), std::mem::replace(text, short.into()).into()));
            }
            let mut schema = Value::Object((*tool.input_schema).clone());
            cut_schema(&mut schema, &mut Vec::new(), max, &name, &mut full);
            if let Value::Object(schema) = schema
                && full.iter().any(|(parameter, _)| !parameter.is_empty())
            {
                tool.input_schema = Arc::new(schema);
            }
            if !full.is_empty() {
                listed.full.insert(name.clone(), full);
            }
        }
        tool.name = name.into();
        listed.tools.push(tool);
    }
    listed
}

/// `tool` without its parameter `argument`, which the hub fills in itself. Also without the
/// `properties` and `required` that leaves empty, as a server's tool that takes nothing else
/// was before a relay added the parameter. A tool without it is passed on as it came.
fn without_argument(tool: &Tool, argument: &str) -> Tool {
    let schema = &tool.input_schema;
    let has = |key| {
        let found = schema.get(key);
        match found {
            Some(Value::Object(members)) => members.contains_key(argument),
            Some(Value::Array(items)) => items.iter().any(|i| i.as_str() == Some(argument)),
            _ => false,
        }
    };
    if !has("properties") && !has("required") {
        return tool.clone();
    }
    let mut schema = (**schema).clone();
    if let Some(Value::Object(properties)) = schema.get_mut("properties") {
        properties.remove(argument);
        if properties.is_empty() {
            schema.remove("properties");
        }
    }
    if let Some(Value::Array(required)) = schema.get_mut("required") {
        required.retain(|name| name.as_str() != Some(argument));
        if required.is_empty() {
            schema.remove("required");
        }
    }
    let mut tool = tool.clone();
    tool.input_schema = Arc::new(schema);
    tool
}

/// The definition of [`DESCRIBE`].
fn describe_tool() -> Tool {
    serde_json::from_value(json!({
        "name": DESCRIBE,
        "description": "Show in full the description of a tool, or of its parameters, \
            that mcp-hub cut short. Such a description ends with a note naming this tool.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "description": "The tool's name, as in the note."}
            },
            "required": ["tool"]
        },
        "annotations": {"readOnlyHint": true}
    }))
    .expect("a valid tool")
}

#[derive(Clone)]
pub struct Hub {
    upstreams: Arc<Vec<Arc<Upstream>>>,
    stats: Stats,
    /// Where results too large to pass on whole are kept for the agent to read.
    results: Arc<PathBuf>,
    /// Tells this hub from one that ran before it, in [`Hub::tools_version`].
    started: u128,
}

/// How much a tool call returned, for the record: characters of text, and bytes of
/// structured content.
fn sizes(response: &CallToolResponse) -> (usize, usize) {
    let CallToolResponse::Complete(result) = response else {
        return (0, 0);
    };
    let texts = result.content.iter().filter_map(|block| block.as_text());
    let structured = result.structured_content.as_ref();
    (
        texts.map(|text| text.text.chars().count()).sum(),
        structured.map_or(0, |value| value.to_string().len()),
    )
}

/// Something the hub itself has to tell the agent about a call: a tool error, which the
/// agent reads, rather than a protocol error.
fn note(text: String) -> CallToolResponse {
    CallToolResponse::Complete(CallToolResult::error(vec![ContentBlock::text(text)]))
}

impl Hub {
    pub fn new(upstreams: Vec<Arc<Upstream>>) -> Self {
        Self::with_stats(upstreams, Stats::off())
    }

    /// A hub that records its tool calls in `stats`.
    pub fn with_stats(upstreams: Vec<Arc<Upstream>>, stats: Stats) -> Self {
        Self {
            upstreams: Arc::new(upstreams),
            stats,
            results: Arc::new(crate::stats::state_dir().join("results")),
            started: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos()),
        }
    }

    /// For an agent that leaves out the servers in `skip`: something that changes whenever
    /// the tools it gets change, and how many servers have yet to say what tools they have.
    pub fn tools_version(&self, skip: &[String]) -> (String, usize) {
        let listed = self
            .upstreams
            .iter()
            .filter(|u| !skip.iter().any(|s| s == u.name()));
        let (mut changes, mut pending) = (0u64, 0);
        for upstream in listed {
            changes = changes.wrapping_add(upstream.tools_generation());
            pending += usize::from(upstream.pending());
        }
        (self.version(changes), pending)
    }

    /// Ask every server for its tools now, all at once: how many it has, or why it failed.
    /// In config order.
    pub async fn status(&self) -> Vec<(String, Result<usize, String>)> {
        let asking = self.upstreams.iter().map(|u| async move {
            let listed = u.list_tools().await;
            (
                u.name().to_owned(),
                listed.map(|t| t.len()).map_err(|e| e.to_string()),
            )
        });
        futures::future::join_all(asking).await
    }

    fn version(&self, changes: u64) -> String {
        format!("{:x}-{changes}", self.started)
    }

    /// Keep results too large to pass on whole in `dir` instead of the usual place.
    pub fn keeping_results_in(mut self, dir: PathBuf) -> Self {
        self.results = Arc::new(dir);
        self
    }

    /// What the server's config says to do to a result before passing it on. Returns, for
    /// the record, the size the result had if it was kept in a file, and how many
    /// characters dropping empty fields saved.
    fn shrink(
        &self,
        upstream: &Upstream,
        tool: &str,
        result: &mut CallToolResult,
    ) -> (Option<usize>, usize) {
        let mut saved = 0;
        if upstream.server.compact_results {
            let before = crate::results::size(result);
            crate::results::compact(result);
            saved = before.saturating_sub(crate::results::size(result));
        }
        let Some(max) = upstream.server.max_result else {
            return (None, saved);
        };
        let name = format!("{}-{tool}", upstream.name());
        let name = name.replace(|c: char| !c.is_ascii_alphanumeric() && c != '-', "_");
        let cut = crate::results::cap(result, max, &self.results, &name);
        (cut, saved)
    }

    pub fn upstreams(&self) -> &[Arc<Upstream>] {
        &self.upstreams
    }

    /// MCP server instructions: what each upstream is for, so clients know when to search its tools.
    pub fn instructions(&self) -> String {
        self.instructions_for(&Client::default())
    }

    fn instructions_for(&self, client: &Client) -> String {
        let mut text = format!(
            "Tools from several MCP servers, named \"<server>{SEP}<tool>\". \
             Look for a server's tools when the task matches it:"
        );
        for u in self.upstreams.iter().filter(|u| !client.skips(u)) {
            text.push_str(&format!("\n- {}", u.name()));
            if let Some(d) = &u.server.description {
                text.push_str(&format!(": {d}"));
            }
        }
        text
    }

    /// Every upstream's tools, from each one's cached list: see [`Upstream::tools`].
    pub async fn list_tools(&self) -> Vec<Tool> {
        self.tools_for(&Client::default()).await.0
    }

    /// The tools `client` gets, and their version (see [`Hub::tools_version`]).
    async fn tools_for(&self, client: &Client) -> (Vec<Tool>, String) {
        let upstreams: Vec<_> = self.upstreams.iter().filter(|u| !client.skips(u)).collect();
        let lists = futures::future::join_all(upstreams.iter().map(|u| u.tools_as_of())).await;
        let count = lists.iter().map(|(l, _)| l.as_ref().map_or(0, |l| l.len()));
        let mut tools = Vec::with_capacity(count.sum::<usize>() + 1);
        let mut changes = 0u64;
        for (upstream, (list, generation)) in upstreams.iter().zip(lists) {
            changes = changes.wrapping_add(generation);
            let list = list.iter().flat_map(|list| list.iter());
            match client.folder_for(upstream) {
                Some((argument, _)) => {
                    tools.extend(list.map(|tool| without_argument(tool, argument)));
                }
                None => tools.extend(list.cloned()),
            }
        }
        let mut listing = self.upstreams.iter().filter(|u| !client.skips(u));
        if listing.any(|u| u.cuts_descriptions()) {
            tools.push(describe_tool());
        }
        (tools, self.version(changes))
    }

    /// Answer a call of [`DESCRIBE`]: what was cut from the tool named in `arguments`.
    async fn describe(
        &self,
        client: &Client,
        arguments: Option<&rmcp::model::JsonObject>,
    ) -> String {
        let asked = arguments
            .and_then(|a| a.get("tool"))
            .and_then(Value::as_str);
        let asked = asked.unwrap_or_default().trim();
        // Agents put a prefix of their own before the hub's names ("mcp__hub__").
        let found = self.upstreams.iter().find_map(|u| {
            let at = asked.find(&format!("{}{SEP}", u.name()))?;
            (at == 0 || asked[..at].ends_with(SEP)).then_some((u, &asked[at..]))
        });
        let Some((upstream, name)) = found.filter(|(u, _)| !client.skips(u)) else {
            return format!(
                "No tool named \"{asked}\". Give the name as in the note, like \"server{SEP}tool\"."
            );
        };
        upstream.tools().await;
        let Some(full) = upstream.full_descriptions(name) else {
            return format!(
                "Nothing was cut from {name}: what its definition says is all there is."
            );
        };
        let mut text = format!("{name}, in full where mcp-hub had cut it.");
        for (parameter, description) in full {
            let title = match parameter.as_str() {
                "" => "Description".to_owned(),
                parameter => format!("Parameter {parameter}"),
            };
            text.push_str(&format!("\n\n## {title}\n\n{description}"));
        }
        text
    }

    /// The agent in `session` is done: stop its copies of the `perAgent` servers.
    pub fn end_session(&self, session: &str) {
        for upstream in self.upstreams.iter() {
            upstream.end_session(session);
        }
    }

    fn route(&self, name: &str) -> Option<(&Arc<Upstream>, String)> {
        let (server, tool) = name.split_once(SEP)?;
        let upstream = self.upstreams.iter().find(|u| u.name() == server)?;
        Some((upstream, tool.to_owned()))
    }
}

impl ServerHandler for Hub {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("mcp-hub", env!("CARGO_PKG_VERSION")))
            .with_instructions(self.instructions())
    }

    // The two ways a client opens a session; both tell it what the servers are for.
    async fn initialize(
        &self,
        request: InitializeRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<InitializeResult, ErrorData> {
        context.peer.set_peer_info(request.clone());
        let mut result = self.negotiate_initialize(&request)?;
        result.instructions = Some(self.instructions_for(&Client::of(&context)));
        Ok(result)
    }

    async fn discover(
        &self,
        context: RequestContext<RoleServer>,
    ) -> Result<DiscoverResult, ErrorData> {
        let mut info = self.get_info();
        info.instructions = Some(self.instructions_for(&Client::of(&context)));
        let versions = self.supported_protocol_versions().into_owned();
        Ok(DiscoverResult::from_server_info(versions, info))
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        let (client, started) = (Client::of(&context), Instant::now());
        let (tools, version) = self.tools_for(&client).await;
        self.stats.record(
            "list",
            json!({
                "client": client.name,
                "skip": client.skipped(),
                "ms": started.elapsed().as_millis() as u64,
                "tools": tools.len(),
            }),
        );
        let mut result = ListToolsResult::with_all_items(tools);
        // Exactly the version of this list, for `mcp-hub connect`; the header may be older.
        let meta = serde_json::Map::from_iter([(TOOLS_META.to_owned(), version.into())]);
        result.meta = Some(rmcp::model::MetaObject(meta));
        Ok(result)
    }

    async fn call_tool(
        &self,
        mut request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let full_name = request.name.to_string();
        if full_name == DESCRIBE {
            let (client, started) = (Client::of(&context), Instant::now());
            let text = self.describe(&client, request.arguments.as_ref()).await;
            self.stats.record(
                "call",
                json!({
                    "server": "hub",
                    "tool": "describe",
                    "client": client.name,
                    "skip": client.skipped(),
                    "ms": started.elapsed().as_millis() as u64,
                    "outcome": "ok",
                    "text": text.chars().count(),
                }),
            );
            let result = CallToolResult::success(vec![ContentBlock::text(text)]);
            return Ok(CallToolResponse::Complete(result));
        }
        let Some((upstream, tool)) = self.route(&full_name) else {
            return Err(ErrorData::invalid_params(
                format!("Unknown tool: {full_name}"),
                None,
            ));
        };
        let (client, started) = (Client::of(&context), Instant::now());
        let (mut cut, mut compacted, mut folded) = (None, 0, false);
        let (outcome, error, response) = if client.skips(upstream) {
            let var = upstream.server.skip_when_env.as_deref();
            let why = var
                .map(|var| format!(" ({var} is set)"))
                .unwrap_or_default();
            let text = format!(
                "mcp-hub: {} is left out of this session{why}. \
                 Use the tools this session has of its own for it.",
                upstream.name()
            );
            ("skipped", None, Ok(note(text)))
        } else {
            request.name = tool.clone().into();
            // Where the agent works, unless it named a folder itself (it was not listed).
            if let Some((argument, folder)) = client.folder_for(upstream) {
                let arguments = request.arguments.get_or_insert_default();
                let given = arguments.get(argument);
                if given.is_none_or(|v| v.is_null() || v.as_str() == Some("")) {
                    arguments.insert(argument.to_owned(), folder.into());
                    folded = true;
                }
            }
            let session = client.session.as_deref();
            match upstream.call_tool_for(request, session, context.ct).await {
                Ok(CallToolResponse::Complete(mut result)) => {
                    (cut, compacted) = self.shrink(upstream, &tool, &mut result);
                    let outcome = match result.is_error {
                        Some(true) => "tool_error",
                        _ => "ok",
                    };
                    (outcome, None, Ok(CallToolResponse::Complete(result)))
                }
                Ok(response) => ("ok", None, Ok(response)),
                // Its message may repeat what was asked: only the code is recorded.
                Err(UpstreamError::Mcp(e)) => {
                    ("rpc_error", Some(format!("code {}", e.code.0)), Err(e))
                }
                Err(e) => {
                    tracing::warn!("[{}] tools/call {full_name} failed: {e}", upstream.name());
                    let text = format!("mcp-hub: {e}");
                    ("failed", Some(e.to_string()), Ok(note(text)))
                }
            }
        };
        let (text, structured) = response.as_ref().map_or((0, 0), sizes);
        self.stats.record(
            "call",
            json!({
                "server": upstream.name(),
                "tool": tool,
                "client": client.name,
                "skip": client.skipped(),
                "ms": started.elapsed().as_millis() as u64,
                "outcome": outcome,
                "error": error.map(|e| e.chars().take(160).collect::<String>()),
                "text": text,
                "structured": structured,
                "cut": cut,
                "compacted": (compacted > 0).then_some(compacted),
                "folder": folded.then_some(true),
            }),
        );
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folders_in_headers() {
        let path = "/home/me/My Projects/café 100%";
        let encoded = encode_folder(path.as_bytes());
        assert_eq!(encoded, "/home/me/My%20Projects/caf%C3%A9%20100%25");
        let mut headers = hyper::HeaderMap::new();
        let mut with = |value: &str| {
            headers.insert(FOLDER_HEADER, value.parse().unwrap());
            folder(&headers)
        };
        assert_eq!(with(&encoded).as_deref(), Some(path));
        assert_eq!(with("relative/path"), None);
        assert_eq!(with("/bad%2"), None);
        assert_eq!(with("/not%FFutf8"), None);
        assert_eq!(with("/nul%00"), None);
    }

    #[test]
    fn tool_names() {
        assert!(usable_tool_name("sentry__search_issues"));
        assert!(!usable_tool_name("x__bad.name"));
        assert!(!usable_tool_name(&"a".repeat(65)));
    }

    #[test]
    fn cuts_long_descriptions() {
        assert_eq!(cut(&"word ".repeat(40), 200, "s__t"), None);
        // Only just too long: the note would take the room that cutting saves.
        assert_eq!(cut(&"word ".repeat(43), 200, "s__t"), None);

        let text = format!(
            "{}\n\n{}",
            "first ".repeat(25).trim(),
            "second ".repeat(100)
        );
        let short = cut(&text, 200, "s__t").unwrap();
        let (head, note) = short.split_once("\n\n[").unwrap();
        assert_eq!(head, "first ".repeat(25).trim());
        assert!(
            note.contains(r#"call hub__describe with {"tool": "s__t"}"#),
            "{note}"
        );
        assert!(
            note.contains(&format!("{} more characters", text.len() - head.len())),
            "{note}"
        );

        // No paragraph or line to end at, and characters of several bytes.
        let short = cut(&"é".repeat(1000), 200, "s__t").unwrap();
        assert!(short.starts_with(&"é".repeat(200)) && !short.starts_with(&"é".repeat(201)));
    }

    #[test]
    fn cuts_tool_and_parameter_descriptions() {
        let long = "x ".repeat(400);
        let tool: Tool = serde_json::from_value(json!({
            "name": "t",
            "description": long,
            "inputSchema": {"type": "object", "properties": {
                "command": {"type": "string", "description": long},
                "description": {"type": "string", "description": "short"},
                "nested": {"type": "object", "properties": {"deep": {"description": long}}},
            }},
        }))
        .unwrap();
        let plain: Tool =
            serde_json::from_value(json!({"name": "p", "inputSchema": {"type": "object"}}))
                .unwrap();
        let all = [tool.clone(), plain.clone()];

        let listed = hub_tools("s", &all, Some(200));
        let parts: Vec<_> = listed.full["s__t"]
            .iter()
            .map(|(p, _)| p.as_str())
            .collect();
        assert_eq!(parts, ["", "command", "nested.deep"]);
        assert!(listed.full["s__t"].iter().all(|(_, text)| *text == long));
        let cut_tool = &listed.tools[0];
        assert!(cut_tool.description.as_ref().unwrap().len() < 400);
        let schema = Value::Object((*cut_tool.input_schema).clone());
        assert!(
            schema["properties"]["command"]["description"]
                .as_str()
                .unwrap()
                .len()
                < 400
        );
        assert_eq!(schema["properties"]["description"]["description"], "short");
        // A tool with nothing to cut is passed on as it came.
        assert!(!listed.full.contains_key("s__p"));
        assert!(Arc::ptr_eq(
            &listed.tools[1].input_schema,
            &plain.input_schema
        ));

        let listed = hub_tools("s", &all, None);
        assert!(listed.full.is_empty());
        assert_eq!(listed.tools[0].description, tool.description);
    }
}
