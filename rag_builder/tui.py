#!/usr/bin/env python3
"""
camel_tui.py
A beautifully designed single-screen CAMEL RAG workstation with enhanced aesthetics.
"""

import os
import shlex
import asyncio
import itertools
from datetime import datetime
from typing import Iterable, List
import uuid

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, Completion, WordCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, HSplit, VSplit, Window, Float, FloatContainer, Dimension
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.widgets import TextArea, Frame
from prompt_toolkit.styles import Style
from qdrant_client import models

from rag_builder.pipeline import RAGPipeline
from rag_builder.utils import read_rags_config, write_rags_config

# -----------------------------
# Enhanced Aesthetic Config
# -----------------------------
# -----------------------------
# Enhanced Aesthetic Config
# -----------------------------
style = Style.from_dict({
    # Main UI elements
    "title": "bold #ffffff bg:#2d1b69",
    "subtitle": "#a78bfa italic",
    "status.ready": "bold #10b981 bg:#064e3b",
    "status.working": "bold #f59e0b bg:#451a03",
    "status.error": "bold #ef4444 bg:#450a0a",
    "loading": "bold #06b6d4",
    
    # Log styling
    "log.default": "#e5e7eb",
    "log.success": "#10b981",
    "log.error": "#ef4444",
    "log.warning": "#f59e0b",
    "log.info": "#3b82f6",
    "log.prompt": "#a78bfa bold",
    
    # Frames and borders
    "frame.title": "bold #ffffff bg:#374151",
    "frame.border": "#6b7280",
    "frame.focused": "#8b5cf6",
    "frame.command": "#10b981",
    
    # Input styling
    "input": "#ffffff bg:#1f2937",
    "input.focused": "#ffffff bg:#111827",
    "input.placeholder": "#6b7280 italic",
    
    # Completion menu
    "completion-menu": "",
    "completion-menu.completion": "fg:#888888",
    "completion-menu.completion.current": "fg:#a78bfa bold",
    "completion-menu.meta": "#9ca3af",
    
    # Scrollbar
    "scrollbar.background": "bg:#374151",
    "scrollbar.button": "bg:#6b7280",
    
    # Decorative elements
    "separator": "#4b5563",
    "accent": "#8b5cf6",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
})

# ... (rest of the file is the same until create_main_layout)

# -----------------------------
# Enhanced Layout and Application
# -----------------------------
def create_main_layout():
    # Create input window with enhanced styling
    input_window = Window(
        BufferControl(buffer=input_buffer),
        height=1,
        style="class:input"
    )
    
    # Create frames with enhanced styling
    log_frame = Frame(
        body=log_area,
        title="📋 Activity Log",
        style="class:frame.border"
    )
    
    input_frame = Frame(
        body=input_window,
        title="",
        style="class:frame.command"
    )
    
    stats_frame = Frame(
        body=stats_display,
        title="📊 Stats",
        style="class:frame.border",
        width=Dimension(preferred=40)
    )
    
    # Create the main content area that will expand
    main_area = VSplit([
        log_frame,
        Window(width=1, char='│', style="class:separator"),
        stats_frame,
    ])

    # The root container
    root_container = HSplit([
        create_title_bar(),
        create_separator(),
        main_area, # This will now expand
        create_status_bar(),
        input_frame,
        completions_window
    ])
    
    return Layout(root_container, focused_element=input_window)

# Enhanced UI symbols
SYMBOLS = {
    'success': '✓',
    'error': '✗',
    'warning': '⚠',
    'info': 'ℹ',
    'arrow': '→',
    'bullet': '•',
    'spinner': ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'],
    'box': ['┌', '┐', '└', '┘', '─', '│'],
}

# -----------------------------
# Enhanced UI Elements
# -----------------------------
def create_title_bar():
    return Window(
        content=FormattedTextControl([
            ("class:title", "🐪 CAMEL RAG Workstation"),
            ("class:subtitle", " • Retrieval-Augmented Generation Interface")
        ]),
        height=1,
        style="class:title"
    )

def create_status_bar():
    global status_text, status_style
    status_text = "Ready"
    status_style = "class:status.ready"  # Add 'class:' prefix
    
    def get_status_content():
        return [
            (status_style, f" {status_text} "),
        ]
    
    return Window(
        content=FormattedTextControl(get_status_content),
        height=1
    )

def create_separator():
    return Window(
        char='─',
        height=1,
        style="class:separator"
    )

# Enhanced log area with syntax highlighting
log_area = TextArea(
    style="class:log.default",
    scrollbar=True,
    focusable=False,
    wrap_lines=True,
    read_only=False  # Allow editing so we can add log messages
)

# Input buffer with enhanced styling
input_buffer = Buffer(
    multiline=False
)

# Stats display
stats_display = TextArea(
    text="",
    style="class:log.info",
    scrollbar=False,
    focusable=False,
    read_only=False,  # Allow editing so we can update stats
)

# -----------------------------
# Enhanced Utilities / Logging
# -----------------------------
def log(msg: str, level: str = "default"):
    try:
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Style the message based on level
        if level == "success":
            formatted_msg = f"[{timestamp}] {SYMBOLS['success']} {msg}"
        elif level == "error":
            formatted_msg = f"[{timestamp}] {SYMBOLS['error']} {msg}"
        elif level == "warning":
            formatted_msg = f"[{timestamp}] {SYMBOLS['warning']} {msg}"
        elif level == "info":
            formatted_msg = f"[{timestamp}] {SYMBOLS['info']} {msg}"
        elif level == "prompt":
            formatted_msg = f"[{timestamp}] {SYMBOLS['arrow']} {msg}"
        else:
            formatted_msg = f"[{timestamp}] {SYMBOLS['bullet']} {msg}"
        
        # Append to existing text instead of using buffer methods
        current_text = log_area.text
        log_area.text = current_text + formatted_msg + "\n"
        
        # Move cursor to end
        log_area.buffer.cursor_position = len(log_area.buffer.text)
    except Exception as e:
        # Fallback logging if main logging fails
        try:
            current_text = log_area.text
            log_area.text = current_text + f"[LOG ERROR] {msg}\n"
        except:
            pass

active_rag_name = None

async def update_stats():
    """Update the stats display with current system information"""
    global rags_config, active_rag_name
    try:
        rag_count_str = f"RAG Indexes: {len(rags_config)}"
        active_rag_str = f"Active RAG: {active_rag_name}" if active_rag_name else "Active RAG: None"
        
        stats_lines = [rag_count_str, active_rag_str]
        
        if active_rag_name:
            selected_rag = next((rag for rag in rags_config if rag['name'] == active_rag_name), None)
            if selected_rag:
                collection_name = selected_rag['collection_name']
                stats = await asyncio.to_thread(pipeline.get_collection_stats, collection_name)
                stats_lines.append("---")
                stats_lines.append(f"Documents: {stats.get('documents', 'N/A')}")
                stats_lines.append(f"Vectors: {stats.get('vectors', 'N/A')}")

        stats_display.text = "\n".join(stats_lines)
    except Exception as e:
        stats_display.text = "Error updating stats."

from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.filters import is_done
# ... (imports)

# -----------------------------
# Completion State Manager
# -----------------------------
class CompletionManager:
    def __init__(self, completer, buffer):
        self.completer = completer
        self.buffer = buffer
        self.completions = []
        self.selected_index = 0
        self.active = False
        self.buffer.on_text_changed += self._on_text_changed

    def _on_text_changed(self, buffer):
        if not self.buffer.text:
            self.active = False
            return
        
        self.completions = list(self.completer.get_completions(
            self.buffer.document, None
        ))
        
        if self.completions:
            self.active = True
            self.selected_index = 0
        else:
            self.active = False

    def get_formatted_completions(self):
        result = []
        for i, c in enumerate(self.completions):
            style = "class:completion-menu.completion.current" if i == self.selected_index else "class:completion-menu.completion"
            result.append((style, f" {c.text} ", "class:completion-menu.meta", f" {c.display_meta} \n"))
        return result

    def next(self):
        if self.completions:
            self.selected_index = (self.selected_index + 1) % len(self.completions)

    def previous(self):
        if self.completions:
            self.selected_index = (self.selected_index - 1 + len(self.completions)) % len(self.completions)

    def apply_completion(self):
        if self.completions:
            comp = self.completions[self.selected_index]
            self.buffer.text = comp.text
            self.buffer.cursor_position = len(self.buffer.text)
            self.active = False


# -----------------------------
# Custom Completers (Enhanced)
# -----------------------------
class PathCompleter(Completer):

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        try:
            path_to_complete = document.text_before_cursor
            
            # Determine the directory and the prefix to match
            if os.path.isdir(path_to_complete):
                base_dir = path_to_complete
                prefix = ""
            else:
                base_dir = os.path.dirname(path_to_complete) or "."
                prefix = os.path.basename(path_to_complete)

            # List entries in the directory
            for entry in sorted(os.listdir(base_dir)):
                if entry.startswith(prefix):
                    full_path = os.path.join(base_dir, entry)
                    
                    # Offer completion
                    if os.path.isdir(full_path):
                        yield Completion(
                            entry + os.sep, 
                            start_position=-len(prefix), 
                            display_meta="📁 directory"
                        )
                    else:
                        ext = os.path.splitext(entry)[1].lower()
                        icon = "📄"
                        if ext in ['.py', '.js', '.html']:
                            icon = "💻"
                        yield Completion(
                            entry, 
                            start_position=-len(prefix), 
                            display_meta=f"{icon} file"
                        )
        except OSError:
            pass

class CommandCompleter(Completer):
    def __init__(self):
        global rags_config  # Add global reference
        self.commands = {
            '/create': '🔨 Create a new RAG index',
            '/list': '📋 List all RAG indexes',
            '/ingest': '📥 Ingest a file, directory, or URL',
            '/ask': '❓ Query a RAG index',
            '/set': '🔧 Set the active RAG index',
            '/stats': '📊 Show system statistics',
            '/clear': '🧹 Clear the log',
            '/quit': '🚪 Exit application',
            '/help': '❓ Show help information'
        }
        self.main_commands = WordCompleter(
            list(self.commands.keys()), 
            ignore_case=True,
            meta_dict=self.commands
        )
        self.path_completer = PathCompleter()

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        words = text.split()
        
        if len(words) == 0 or (len(words) == 1 and not text.endswith(' ')):
            yield from self.main_commands.get_completions(document, complete_event)
            return

        command = words[0]
        
        if command == '/set':
            if len(words) == 1 and text.endswith(' '):
                index_names = [rag['name'] for rag in rags_config]
                for name in index_names:
                    yield Completion(name, start_position=0, display_meta="🗃️ RAG index")
            elif len(words) == 2 and not text.endswith(' '):
                index_names = [rag['name'] for rag in rags_config]
                for name in index_names:
                    if name.startswith(words[1]):
                        yield Completion(name, start_position=-len(words[1]), display_meta="🗃️ RAG index")
        elif command == '/ingest' and len(words) > 1:
            path_text = " ".join(words[1:])
            path_doc = Document(path_text, cursor_position=len(path_text))
            yield from self.path_completer.get_completions(path_doc, complete_event)
        else:
            yield from self.main_commands.get_completions(document, complete_event)


# -----------------------------
# Key bindings and behaviors
# -----------------------------
kb = KeyBindings()

@kb.add("c-c")
def exit_app(event):
    """Enhanced exit with confirmation"""
    log("Goodbye! 👋", "info")
    event.app.exit()

@kb.add("enter")
def handle_enter(event):
    """Enhanced enter handling with better feedback"""
    text = input_buffer.text.strip()
    if text:
        log(text, "prompt")
        input_buffer.text = ""  # Clear input buffer properly
        completion_manager.active = False
        event.app.create_background_task(handle_command(text))

@kb.add("c-l")
def clear_screen(event):
    """Clear the log area"""
    log_area.text = ""
    log("Screen cleared", "info")

@kb.add("down")
def _(event):
    completion_manager.next()

@kb.add("up")
def _(event):
    completion_manager.previous()

@kb.add("tab")
def _(event):
    completion_manager.apply_completion()

# -----------------------------
# Enhanced Spinner Control
# -----------------------------
spinner_running = False
_spinner_task = None

async def spinner_loop():
    """Enhanced spinner with better animations"""
    global spinner_running, status_text, status_style
    spinner_running = True
    spinner_chars = SYMBOLS['spinner']
    cycle = itertools.cycle(spinner_chars)
    original_text = status_text
    
    try:
        while spinner_running:
            spinner_char = next(cycle)
            status_text = f"{spinner_char} {original_text}"
            status_style = "class:status.working"  # Add 'class:' prefix
            app.invalidate()  # Force a redraw
            await asyncio.sleep(0.1)
    finally:
        status_text = "Ready"
        status_style = "class:status.ready"  # Add 'class:' prefix
        app.invalidate()  # Final redraw to reset the status


def start_spinner(action_text: str):
    """Start spinner with action text"""
    global _spinner_task, status_text
    status_text = action_text
    if _spinner_task is None or _spinner_task.done():
        _spinner_task = asyncio.create_task(spinner_loop())

def stop_spinner():
    """Stop spinner and reset status"""
    global spinner_running, _spinner_task, status_text, status_style
    spinner_running = False
    if _spinner_task:
        _spinner_task.cancel()
    _spinner_task = None
    status_text = "Ready"
    status_style = "class:status.ready"  # Add 'class:' prefix

# -----------------------------
# Enhanced Command handling
# -----------------------------
pipeline = RAGPipeline()
rags_config = read_rags_config()

async def handle_command(cmd: str):
    global rags_config
    
    # Manual parsing for ingest command
    if cmd.strip().startswith('/ingest '):
        parts = cmd.strip().split(maxsplit=1)
        command = parts[0]
        args = [parts[1]] if len(parts) > 1 else []
    else:
        try:
            parts = shlex.split(cmd, posix=False)
            command = parts[0]
            args = parts[1:]
        except (ValueError, IndexError):
            log("Invalid command syntax", "error")
            return

    if command == '/create':
        if len(args) != 1:
            log("Usage: /create <index_name>", "warning")
            return
        name = args[0]
        if any(rag['name'] == name for rag in rags_config):
            log(f"Index '{name}' already exists", "error")
            return
        
        start_spinner(f"Creating index '{name}'...")
        
        collection_name = str(uuid.uuid4())
        
        # Attempt to create the collection in Qdrant
        try:
            # The vector size for all-MiniLM-L6-v2 is 384
            pipeline.qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
            )
            log(f"Qdrant collection '{collection_name[:8]}...' created.", "success")
        except Exception as e:
            stop_spinner()
            log(f"Failed to create Qdrant collection: {e}", "error")
            return

        new_rag = {"name": name, "collection_name": collection_name}
        rags_config.append(new_rag)
        write_rags_config(rags_config)
        
        stop_spinner()
        log(f"RAG index '{name}' created successfully", "success")
        await update_stats()

    elif command == '/list':
        if not rags_config:
            log("No RAG indexes found", "warning")
            return
        
        log("Available RAG indexes:", "info")
        for i, rag in enumerate(rags_config, 1):
            log(f"  {i}. {rag['name']} (ID: {rag['collection_name'][:8]}...)")

    elif command == '/set':
        if len(args) != 1:
            log("Usage: /set <index_name>", "warning")
            return
        name = args[0]
        if any(rag['name'] == name for rag in rags_config):
            global active_rag_name
            active_rag_name = name
            log(f"Active RAG index set to '{name}'", "success")
        else:
            log(f"RAG index '{name}' not found", "error")

    elif command == '/ingest':
        if not active_rag_name:
            log("No active RAG index. Use /set <index_name> to set one.", "error")
            return
        if len(args) != 1:
            log("Usage: /ingest <path_or_url>", "warning")
            return
        
        name, path = active_rag_name, args[0]
        selected_rag = next((rag for rag in rags_config if rag['name'] == name), None)
        
        # This check is now redundant if /set only allows valid names, but good for safety
        if not selected_rag:
            log(f"Active RAG index '{name}' not found. Please set a valid index.", "error")
            return
        
        # If it's not a URL, check if the path exists
        if not (path.startswith("http://") or path.startswith("https://")):
            if not os.path.exists(path):
                log(f"Path '{path}' does not exist", "error")
                return
        
        start_spinner(f"Ingesting '{path}' into '{name}'")
        success = await asyncio.to_thread(pipeline.ingest, path, selected_rag['collection_name'], logger=log)
        stop_spinner()
        if success:
            log(f"Finished ingesting data into '{name}'.", "success")

    elif command == '/ask':
        if not active_rag_name:
            log("No active RAG index. Use /set <index_name> to set one.", "error")
            return
        if not args:
            log("Usage: /ask \"<query>\"", "warning")
            return

        name = active_rag_name
        query = " ".join(args)
        selected_rag = next((rag for rag in rags_config if rag['name'] == name), None)

        # This check is now redundant if /set only allows valid names, but good for safety
        if not selected_rag:
            log(f"Active RAG index '{name}' not found. Please set a valid index.", "error")
            return
        
        start_spinner(f"Querying '{name}' with: {query[:30]}{'...' if len(query) > 30 else ''}")
        await pipeline.ask(query, selected_rag['collection_name'], logger=log)
        stop_spinner()
        log(f"Query completed for '{name}'", "success")

    elif command == '/stats':
        log("System Statistics:", "info")
        log(f"  RAG Indexes: {len(rags_config)}")
        log("  Memory Usage: Available") # Could add actual memory stats
        log("  Uptime: Active")

    elif command == '/clear':
        log_area.text = ""
        log("Welcome back to CAMEL RAG Workstation! 🐪", "success")

    elif command == '/quit':
        log("Shutting down gracefully...", "info")
        await asyncio.sleep(0.5)
        app.exit()

    elif command == '/help':
        log("CAMEL RAG Workstation - Command Reference", "info")
        log("=" * 50)
        for cmd, desc in CommandCompleter().commands.items():
            log(f"  {cmd:<12} {desc}")
        log("=" * 50)
        log("Keyboard shortcuts:", "info")
        log("  Ctrl+C       Exit application")
        log("  Ctrl+L       Clear screen")

    else:
        log(f"Unknown command: {command}", "error")
        log("Type /help for available commands", "info")

    await update_stats()

# -----------------------------
# Enhanced Layout and Application
# -----------------------------
def create_main_layout():
    # Create input window with enhanced styling
    input_window = Window(
        BufferControl(buffer=input_buffer),
        height=1,
        style="class:input"
    )
    
    # Create frames with enhanced styling
    log_frame = Frame(
        body=log_area,
        title="📋 Activity Log",
        style="class:frame.border"
    )
    
    input_frame = Frame(
        body=input_window,
        title="",
        style="class:frame.command"
    )
    
    stats_frame = Frame(
        body=stats_display,
        title="📊 Stats",
        style="class:frame.border"
    )
    
    # Create the main content area that will expand
    main_area = VSplit([
        log_frame,
        Window(width=1, char='│', style="class:separator"),
        stats_frame,
    ])

    # The root container
    root_container = HSplit([
        create_title_bar(),
        create_separator(),
        main_area, # This will now expand
        create_status_bar(),
        input_frame,
    ])
    
    return Layout(
        FloatContainer(
            content=root_container,
            floats=[
                Float(
                    bottom=1,
                    left=1,
                    right=1,
                    height=8,
                    content=CompletionsMenu(max_height=8),
                )
            ],
        ),
        focused_element=input_window
    )

# Initialize completer
input_buffer = Buffer(
    multiline=False,
    complete_while_typing=True,
)
completion_manager = CompletionManager(completer=CommandCompleter(), buffer=input_buffer)

def get_completions_formatted_text():
    return completion_manager.get_formatted_completions()

completions_window = Window(
    content=FormattedTextControl(get_completions_formatted_text),
    height=8,
)

# Create application
app = Application(
    layout=create_main_layout(),
    key_bindings=kb,
    style=style,
    full_screen=True,
    mouse_support=True
)

async def main():
    """Enhanced startup sequence"""
    global status_text, status_style, rags_config
    
    # Initialize rags_config if not already loaded
    if not rags_config:
        try:
            rags_config = read_rags_config()
        except:
            rags_config = []
    
    # Initialize status with proper class prefix
    status_text = "Ready"
    status_style = "class:status.ready"
    
    # Initialize
    await pipeline.async_init()
    await update_stats()
    
    # Welcome message
    log("🐪 Welcome to CAMEL RAG Workstation", "success")
    log("Advanced Retrieval-Augmented Generation Interface", "info")
    log("Type /help for command reference", "info")
    log("─" * 50)
    
    # Start the application
    try:
        await app.run_async()
    except Exception as e:
        log(f"An unexpected error occurred: {e}", "error")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("🐪 CAMEL RAG Workstation closed. Goodbye!")

