"""Error types for the tool server client.

Imports neither `slicer` nor `qt` - see ARCHITECTURE.md dependency rule.
"""

import html


def plain_text(text) -> str:
    """`text` as Qt will show it LITERALLY, in a message box or a tooltip.

    Both render "rich text" whenever Qt guesses the string is HTML, and most of
    what they show here came from the server -- an error's `detail`, a text
    tool's answer, a run's terminal message, a schema's description. Shown as
    HTML, that is markup a server controls in a dialog with clickable links:
    a convincing "click here to re-authenticate". Escaped, it is just text.

    A string with no `<` is returned unchanged: Qt never takes it for HTML, and
    escaping it would print `&amp;` literally. Anything else is escaped and
    wrapped in `<qt>`, which makes Qt render it as (inert) rich text, so the
    entities read back as the characters they stand for; line breaks are kept.
    """
    text = "" if text is None else str(text)
    if "<" not in text:
        return text
    escaped = html.escape(text).replace("\n", "<br>")
    return f"<qt><p style='white-space: pre-wrap'>{escaped}</p></qt>"


def preformatted_text(text) -> str:
    """`text` as Qt will show it in a monospaced block, every space kept.

    For a folder tree: Qt shows a plain-text tooltip in the proportional UI
    font and collapses runs of spaces, so the indentation that IS a tree came
    out flat. Escaped first, exactly as `plain_text` escapes, so whatever the
    server sent is still inert -- only the `<pre>` around it is markup.
    """
    text = "" if text is None else str(text)
    return f"<qt><pre>{html.escape(text)}</pre></qt>"


class ServerToolError(Exception):
    """A presentable failure while talking to the tool server."""

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class RunCancelled(ServerToolError):
    """The run was withdrawn by whoever asked for it (HTTP 499).

    A class of its own, and that is the whole point of the status code being
    non-standard: no standard code means "the caller withdrew this", and the
    difference between a cancellation and a failure is the difference between
    closing a panel quietly and opening an error dialog. A caller that does not
    know this class still gets a ServerToolError and behaves as it always did.
    """


def error_for_status(status_code, server_message=None):
    """Map an HTTP status code to a ServerToolError with a user-facing message.

    The server's own `detail` is shown verbatim whenever it sends one - it is
    consistently the more specific message, and for the argument errors it is
    the only one that can be: 422 names the offending argument and, for an
    option outside a `choice`/`multichoice` list, spells out what was expected
    ("Argument 'preview_format': unknown option 'xml'. Expected one of: csv,
    json"). Nothing here second-guesses or rewrites it. The strings below are
    fallbacks for a response with no usable body.

    500 is the exception: a crash inside a tool is not a message meant for the
    user, and its detail may leak server-side internals.
    """
    if status_code == 401:
        return ServerToolError(server_message or "Authentication failed.", status_code)
    if status_code == 404:
        # The server distinguishes two very different 404s: a name that does
        # not exist, and a tool whose module failed to import at startup and
        # was therefore skipped by the registry ("Tool 'ALI' failed to load at
        # server startup and is unavailable."). Only its detail can tell them
        # apart, and swallowing it sends the user hunting for a typo when the
        # real cause is a missing server-side dependency.
        return ServerToolError(
            server_message or "Unknown tool. Refresh the tool list and check the name.",
            status_code,
        )
    if status_code == 422:
        return ServerToolError(server_message or "Invalid arguments.", status_code)
    if status_code == 400:
        return ServerToolError(server_message or "Disallowed file type.", status_code)
    if status_code == 413:
        return ServerToolError(server_message or "File too large for the server.", status_code)
    if status_code == 500:
        return ServerToolError("The tool failed on the server.", status_code)
    if status_code == 499:
        # nginx's code, borrowed by the run contract: the client withdrew the
        # request. Not a failure, and it must never reach an error dialog -
        # the user is the one who asked for this.
        return RunCancelled(server_message or "Run cancelled.", status_code)
    return ServerToolError(server_message or f"Server error ({status_code}).", status_code)
