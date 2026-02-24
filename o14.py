import gradio as gr
import requests
import json
import os
import datetime
import textwrap
import random
import threading
import webbrowser
import shutil
import time
import base64
import hashlib
import re
from pathlib import Path

# =========================================================
# --- UI UTILITY FUNCTIONS ---
# =========================================================
def focus_textbox_js():
    """JavaScript to focus the user input textbox after streaming completes."""
    return """
    () => {
        // Find the user input textarea and focus it
        const textareas = document.querySelectorAll('textarea');
        for (let ta of textareas) {
            // Look for the main input textarea (not the chatbot messages)
            if (ta.placeholder && ta.placeholder.includes('Type your message')) {
                ta.focus();
                // Also scroll to bottom of chat
                const chatbot = document.querySelector('.gradio-chatbot');
                if (chatbot) {
                    chatbot.scrollTop = chatbot.scrollHeight;
                }
                break;
            }
        }
    }
    """

# =========================================================
# --- VERSIONING ---
# =========================================================
VERSION = "3.3-schemafix"

# =========================================================
# --- CONFIGURATION & DATA PERSISTENCE ---
# =========================================================
PROMPTS_FILE = "prompts.json"
PARAMS_FILE = "params.json"
HISTORY_FILE = "openrouter_history.json"
THEME_FILE = "theme.json"
LOG_FILE = "chat_log.txt"
DEFAULT_MODEL = "tngtech/deepseek-r1t2-chimera:free"
MAX_BACKUPS = 5

# Context management constants
MAX_CONTEXT_TOKENS = 120000  # Leave room for response
TOKEN_BUFFER = 4000  # Safety buffer
SUMMARY_TRIGGER_TOKENS = 80000  # When to start summarizing
COMPRESSION_LEVELS = 5  # Number of compression tiers

# Default structures remain the same...
DEFAULT_PROMPT = {
    "name": "Default Assistant",
    "content": "You are a helpful, creative, and friendly AI assistant.",
    "active": True,
    "group": "default"
}

DEFAULT_PARAMS = {
    "name": "Default",
    "api_key": "",
    "model": DEFAULT_MODEL,
    "model_vision": "",       # override model for vision/image-input requests
    "model_image_gen": "",    # override model for image-generation requests
    "temperature": 0.7,
    "max_tokens": 1024,
    "response_tokens": 0,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "stream": True,
    "top_k": 0,
    "min_p": 0.0,
    "top_a": 0.0,
    "repetition_penalty": 1.0,
    "seed": -1
}

# Model groups: named sets mapping modalities to specific model IDs
# Stored in params_data["model_groups"] = { "GroupName": {"primary": "...", "vision": "...", "image_gen": "..."} }
DEFAULT_MODEL_GROUPS = {}
MODEL_GROUP_FILE = "model_groups.json"

DEFAULT_THEME = {
    "name": "Dark",
    "primary_color": "#3b82f6",
    "secondary_color": "#64748b",
    "background_color": "#0f172a",
    "surface_color": "#1e293b",
    "text_color": "#f1f5f9",
    "border_color": "#334155",
    "accent_color": "#8b5cf6"
}

THEMES = {
    "Dark": DEFAULT_THEME,
    "Light": {
        "name": "Light",
        "primary_color": "#2563eb",
        "secondary_color": "#64748b",
        "background_color": "#ffffff",
        "surface_color": "#f8fafc",
        "text_color": "#1e293b",
        "border_color": "#e2e8f0",
        "accent_color": "#7c3aed"
    },
    "Midnight": {
        "name": "Midnight",
        "primary_color": "#6366f1",
        "secondary_color": "#475569",
        "background_color": "#020617",
        "surface_color": "#0f172a",
        "text_color": "#e2e8f0",
        "border_color": "#1e293b",
        "accent_color": "#a855f7"
    },
    "Forest": {
        "name": "Forest",
        "primary_color": "#22c55e",
        "secondary_color": "#65a30d",
        "background_color": "#052e16",
        "surface_color": "#14532d",
        "text_color": "#dcfce7",
        "border_color": "#166534",
        "accent_color": "#84cc16"
    },
    "Ocean": {
        "name": "Ocean",
        "primary_color": "#06b6d4",
        "secondary_color": "#0891b2",
        "background_color": "#083344",
        "surface_color": "#164e63",
        "text_color": "#cffafe",
        "border_color": "#155e75",
        "accent_color": "#22d3ee"
    },
    "Sunset": {
        "name": "Sunset",
        "primary_color": "#f97316",
        "secondary_color": "#ea580c",
        "background_color": "#431407",
        "surface_color": "#7c2d12",
        "text_color": "#ffedd5",
        "border_color": "#9a3412",
        "accent_color": "#fb923c"
    }
}


# =========================================================
# --- Globals ---
# =========================================================
last_logged_index = 0
active_session_name = None
_last_save_time = 0.0
_save_interval = 0.5
current_params = {}
active_prompts = []
current_theme = DEFAULT_THEME.copy()
uploaded_files = []
prompts_data = {"prompts": [DEFAULT_PROMPT], "groups": ["default"]}
params_data = {"presets": {"Default": DEFAULT_PARAMS}, "active_preset": "Default"}

_cached_models = None
_models_last_fetch = 0
_model_capabilities_cache = {}
model_groups_data = {}  # { "GroupName": {"primary": "...", "vision": "...", "image_gen": "..."} }

log_as_txt_enabled = False
last_logged_model = None
stop_event = threading.Event()
_inference_active = False


# =========================================================
# --- TOKEN UTILITY ---
# =========================================================
def estimate_tokens(text):
    if not text:
        return 0
    # More accurate estimation: ~4 chars per token for English, less for code
    code_chars = len(re.findall(r'[{}();=+\-*/]', text))
    adjusted_len = len(text) - code_chars // 2
    return max(1, adjusted_len // 4)

def calculate_total_prompt_tokens():
    total = 0
    for prompt in active_prompts:
        if prompt.get("active", False):
            total += estimate_tokens(prompt.get("content", ""))
    return total

# =========================================================
# --- FILE OPERATIONS ---
# =========================================================
def load_json(file_path, default_data=None):
    if default_data is None:
        default_data = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not parse {file_path}: {e}, using default.")
            return default_data
    return default_data

def save_json(file_path, data):
    create_rolling_backup(file_path)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        print(f"Error saving {file_path}: {e}")

def create_rolling_backup(file_path, max_backups=MAX_BACKUPS):
    if not os.path.exists(file_path):
        return
    oldest_backup = f"{file_path}.b{max_backups}"
    if os.path.exists(oldest_backup):
        os.remove(oldest_backup)
    for i in range(max_backups - 1, 0, -1):
        src = f"{file_path}.b{i}"
        dst = f"{file_path}.b{i+1}"
        if os.path.exists(src):
            try:
                os.rename(src, dst)
            except OSError as e:
                print(f"Warning: Could not rename {src} to {dst}: {e}")
    newest_backup = f"{file_path}.b1"
    try:
        shutil.copy2(file_path, newest_backup)
    except IOError as e:
        print(f"Error creating backup {newest_backup}: {e}")


# =========================================================
# --- LOGGING FUNCTIONALITY ---
# =========================================================
def log_assistant_output(assistant_text, model_name):
    global last_logged_model
    if not log_as_txt_enabled or not assistant_text:
        return

    try:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            if last_logged_model != model_name:
                f.write(f"\n[{timestamp}] Model: {model_name}\n")
                f.write("=" * 50 + "\n")
                last_logged_model = model_name
            f.write(f"\n{assistant_text}\n")
            f.write("\n-----------------\n")
    except Exception as e:
        print(f"Error logging to file: {e}")

def set_log_enabled(enabled):
    global log_as_txt_enabled
    log_as_txt_enabled = enabled
    return f"TXT logging {'enabled' if enabled else 'disabled'}."

# =========================================================
# --- INITIALIZATION ---
# =========================================================
def initialize_data():
    global current_params, active_prompts, current_theme, prompts_data, params_data, model_groups_data

    prompts_data = load_json(PROMPTS_FILE, {"prompts": [DEFAULT_PROMPT], "groups": ["default"]})
    all_prompts = prompts_data.get("prompts", [DEFAULT_PROMPT])
    active_prompts = [p for p in all_prompts if p.get("active", False)]
    if not active_prompts:
        active_prompts = [DEFAULT_PROMPT]

    params_data = load_json(PARAMS_FILE, {"presets": {"Default": DEFAULT_PARAMS}, "active_preset": "Default"})
    active_preset_name = params_data.get("active_preset", "Default")
    if active_preset_name in params_data.get("presets", {}):
        current_params = params_data["presets"][active_preset_name].copy()
    else:
        current_params = DEFAULT_PARAMS.copy()

    model_groups_data = load_json(MODEL_GROUP_FILE, {})

    theme_data = load_json(THEME_FILE, {"active_theme": "Dark"})
    theme_name = theme_data.get("active_theme", "Dark")
    current_theme = THEMES.get(theme_name, DEFAULT_THEME).copy()

    return prompts_data, params_data

prompts_data, params_data = initialize_data()
chat_history = load_json(HISTORY_FILE, [])
if not isinstance(chat_history, list):
    chat_history = []


# =========================================================
# --- STOP CONTROL ---
# =========================================================
def set_stop_event():
    global _inference_active
    stop_event.set()
    _inference_active = False
    return "Stopping inference..."

def clear_stop_event():
    global _inference_active
    stop_event.clear()
    _inference_active = True

def is_stop_requested():
    return stop_event.is_set()

def quit_app():
    print("Quitting application...")
    threading.Timer(0.1, os._exit, args=[0]).start()
    return gr.Markdown(value="👋 Shutting down...", visible=True), gr.Button(interactive=False)


# =========================================================
# --- OPENROUTER API ---
# =========================================================
def fetch_openrouter_models():
    global _cached_models, _models_last_fetch
    if _cached_models and (time.time() - _models_last_fetch) < 300:
        return _cached_models
    try:
        response = requests.get("https://openrouter.ai/api/v1/models", timeout=30)
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            _cached_models = models
            _models_last_fetch = time.time()
            return models
    except Exception as e:
        print(f"Error fetching models: {e}")
    return _cached_models or []

def get_modality_codes(model):
    codes = []
    architecture = model.get("architecture", {})
    # New schema: input_modalities is a list e.g. ["image", "text"]
    # Old schema fallback: modality was a string e.g. "text+image->text"
    input_mods = architecture.get("input_modalities", [])
    output_mods = architecture.get("output_modalities", [])
    old_modality = architecture.get("modality", "")

    has_image_input = "image" in input_mods or "text+image" in old_modality or "image+text" in old_modality
    has_image_output = "image" in output_mods or "text->image" in old_modality
    has_text = "text" in input_mods or "text" in old_modality or not input_mods

    if has_text and not has_image_input:
        codes.append("TT")
    if has_image_input:
        codes.append("IT")
    if has_image_output:
        codes.append("TI")
    if model.get("top_provider", {}).get("is_moderated"):
        codes.append("MD")
    return codes if codes else ["TT"]

def get_model_modality_info(model_id):
    global _model_capabilities_cache
    if model_id in _model_capabilities_cache:
        return _model_capabilities_cache[model_id]

    models = fetch_openrouter_models()
    for model in models:
        if model.get("id") == model_id:
            architecture = model.get("architecture", {})
            # New schema uses input_modalities list; fall back to old modality string
            input_mods = architecture.get("input_modalities", [])
            old_modality = architecture.get("modality", "").lower()
            supports_vision = (
                "image" in input_mods or
                "image" in old_modality or
                "vision" in old_modality or
                "multimodal" in old_modality
            )
            info = {
                "supports_vision": supports_vision,
                "supports_text": True,
                "input_modalities": input_mods,
                "context_length": model.get("context_length", 4096),
                "supported_parameters": model.get("supported_parameters", []),
                "found_in_list": True
            }
            _model_capabilities_cache[model_id] = info
            return info

    # Not found — unknown, don't assume no vision
    default_info = {
        "supports_vision": None,
        "supports_text": True,
        "input_modalities": [],
        "context_length": 4096,
        "supported_parameters": [],
        "found_in_list": False
    }
    _model_capabilities_cache[model_id] = default_info
    return default_info

def get_current_model_modality_display():
    """Return a short string describing what the active model(s) support."""
    primary = current_params.get("model", "")
    vision = current_params.get("model_vision", "")
    image_gen = current_params.get("model_image_gen", "")

    info = get_model_modality_info(primary)
    sv = info["supports_vision"]
    parts = []
    if sv is True:
        parts.append("👁 Vision")
    elif sv is None:
        parts.append("❓ Vision unknown")
    if info.get("supports_text", True):
        parts.append("💬 Text")
    primary_caps = " + ".join(parts) if parts else "💬 Text"
    if not info.get("found_in_list", True):
        primary_caps += " *(not in model list — capabilities unverified)*"
    # Show supported params count as a quick health indicator
    sp = info.get("supported_parameters", [])
    if sp:
        primary_caps += f" · {len(sp)} params"

    lines = [f"**Primary:** `{primary}` — {primary_caps}"]
    if vision:
        lines.append(f"**Vision override:** `{vision}`")
    if image_gen:
        lines.append(f"**Image-gen override:** `{image_gen}`")
    return "  \n".join(lines)

def get_modality_label(codes):
    """Convert modality codes to human-readable label."""
    labels = []
    if "IT" in codes:
        labels.append("👁 Vision")
    if "TT" in codes and "IT" not in codes:
        labels.append("💬 Text")
    if "TI" in codes:
        labels.append("🎨 ImgGen")
    if "MD" in codes:
        labels.append("🛡 Mod")
    return " ".join(labels) if labels else "💬 Text"

def format_model_for_display(model):
    name = model.get("name", model.get("id", "Unknown"))
    model_id = model.get("id", "")
    pricing = model.get("pricing", {})
    prompt_price = float(pricing.get("prompt", 0) or 0)
    completion_price = float(pricing.get("completion", 0) or 0)
    context_length = model.get("context_length", 0)
    codes = get_modality_codes(model)
    if prompt_price == 0 and completion_price == 0:
        price_str = "FREE"
    else:
        price_str = f"${prompt_price*1e6:.2f}/${completion_price*1e6:.2f}/1M"
    modality_label = get_modality_label(codes)
    ctx_str = f"{context_length//1000}k ctx" if context_length >= 1000 else f"{context_length} ctx"
    return f"{name}  [{modality_label}]  {price_str}  {ctx_str}", model_id

def search_models(query="", sort_by="rate", modality_filter="all"):
    models = fetch_openrouter_models()
    if not models:
        return []
    if query:
        query_lower = query.lower()
        models = [m for m in models if query_lower in m.get("name", "").lower() or query_lower in m.get("id", "").lower()]
    # Modality filter
    if modality_filter == "text":
        models = [m for m in models if "IT" not in get_modality_codes(m) and "TI" not in get_modality_codes(m)]
    elif modality_filter == "vision":
        models = [m for m in models if "IT" in get_modality_codes(m)]
    elif modality_filter == "image_gen":
        models = [m for m in models if "TI" in get_modality_codes(m)]
    if sort_by == "rate":
        models.sort(key=lambda m: (float(m.get("pricing", {}).get("prompt", 1) or 1) + float(m.get("pricing", {}).get("completion", 1) or 1)))
    elif sort_by == "context":
        models.sort(key=lambda m: m.get("context_length", 0), reverse=True)
    elif sort_by == "name":
        models.sort(key=lambda m: m.get("name", m.get("id", "")).lower())
    return [format_model_for_display(m) for m in models]


# =========================================================
# --- FILE UPLOAD HANDLING ---
# =========================================================
def encode_file_to_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding file: {e}")
        return None

def encode_image_resized(file_path, max_dimension=1568, quality=85):
    """
    Encode image to base64, resizing if too large.
    Most vision APIs cap at ~1568px on the long edge and reject very large payloads.
    Falls back to raw encode if PIL not available.
    """
    try:
        from PIL import Image
        import io
        with Image.open(file_path) as img:
            # Convert RGBA/P to RGB for JPEG compatibility
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")
            w, h = img.size
            if max(w, h) > max_dimension:
                scale = max_dimension / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                print(f"[image] resized {w}x{h} → {img.size[0]}x{img.size[1]}")
            buf = io.BytesIO()
            # Always save as JPEG for vision requests (smaller, universally supported)
            img.save(buf, format="JPEG", quality=quality)
            data = base64.b64encode(buf.getvalue()).decode("utf-8")
            size_kb = len(data) * 3 // 4 // 1024
            print(f"[image] encoded {os.path.basename(file_path)} → {size_kb}KB base64")
            return data, "image/jpeg"
    except ImportError:
        # PIL not available — raw encode, preserve original mime type
        print("[image] PIL not available, encoding raw (install Pillow for auto-resize)")
        data = encode_file_to_base64(file_path)
        return data, None
    except Exception as e:
        print(f"[image] encode error: {e}, falling back to raw")
        data = encode_file_to_base64(file_path)
        return data, None

def handle_file_upload(files):
    global uploaded_files
    if not files:
        return None

    # Check the effective vision model (override if set, else primary)
    vision_override = current_params.get("model_vision", "").strip()
    check_model_id = vision_override if vision_override else current_params.get("model", DEFAULT_MODEL)
    modality_info = get_model_modality_info(check_model_id)

    for file_info in files:
        if isinstance(file_info, dict):
            file_path = file_info.get("path", "")
            file_name = file_info.get("name", os.path.basename(file_path))
        else:
            file_path = file_info
            file_name = os.path.basename(file_path)

        if not file_path or not os.path.exists(file_path):
            continue

        ext = os.path.splitext(file_name)[1].lower()

        if ext in [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"]:
            if modality_info["supports_vision"] is False:
                print(f"Warning: Current model does not support vision — image will be sent as filename text.")
            file_type = "image"
            base64_data, detected_mime = encode_image_resized(file_path)
            if base64_data:
                uploaded_files.append({
                    "name": file_name,
                    "type": file_type,
                    "path": file_path,
                    "base64": base64_data,
                    "mime_type": detected_mime,   # may be overridden to jpeg by resizer
                    "extension": ext
                })
            continue  # skip the generic encode below
        elif ext in [".pdf", ".txt", ".md", ".json", ".csv", ".py", ".js", ".html", ".css", ".doc", ".docx"]:
            file_type = "document"
        else:
            file_type = "other"

        base64_data = encode_file_to_base64(file_path)
        if base64_data:
            uploaded_files.append({
                "name": file_name,
                "type": file_type,
                "path": file_path,
                "base64": base64_data,
                "mime_type": None,
                "extension": ext
            })

    return None  # FIXED: Return None since outputs=[] in Gradio event handler

def clear_uploaded_files():
    global uploaded_files
    uploaded_files = []
    return []

def get_file_upload_status():
    vision_override = current_params.get("model_vision", "").strip()
    check_model_id = vision_override if vision_override else current_params.get("model", DEFAULT_MODEL)
    modality_info = get_model_modality_info(check_model_id)

    if not uploaded_files:
        return ""

    file_list = []
    for f in uploaded_files:
        icon = "📄"
        if f['type'] == 'image':
            icon = "🖼️"
            if modality_info["supports_vision"] is False:
                icon = "⚠️🖼️"
        elif f['type'] == 'document':
            icon = "📃"
        file_list.append(f"{icon} {f['name']}")

    return " | ".join(file_list)

def get_file_count():
    return len(uploaded_files)


# =========================================================
# --- RAG SUPPORT ---
# =========================================================
def simple_rag_extract(text, query, max_chunks=3):
    if not text or not query:
        return text
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    if len(chunks) <= max_chunks:
        return text
    query_words = set(query.lower().split())
    scored_chunks = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(query_words & chunk_words)
        scored_chunks.append((score, chunk))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [c for _, c in scored_chunks[:max_chunks]]
    original_order = [c for c in chunks if c in top_chunks]
    return "\n\n".join(original_order)

# =========================================================
# --- SMART CONTEXT COMPRESSION FOR MAIN CHAT ---
# =========================================================

class ChatContextManager:
    """
    Manages main chat context to prevent overrun.
    Keeps full history on disk, intelligent window in context.
    """

    def __init__(self):
        self.full_history_file = "chat_full_history.jsonl"
        self.compression_threshold = 60000  # tokens

    def add_exchange(self, user_msg, assistant_msg, metadata=None):
        """Log complete exchange to disk."""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "user": user_msg,
            "assistant": assistant_msg,
            "metadata": metadata or {},
            "id": hashlib.md5(f"{time.time()}{user_msg[:50]}".encode()).hexdigest()[:12]
        }
        with open(self.full_history_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')

    def build_optimized_history(self, current_history, new_user_input="", max_tokens=80000):
        """
        Build history that fits in context window.
        Strategy: Keep recent exchanges full, summarize older ones.
        """
        total_tokens = estimate_tokens(new_user_input)
        optimized = []

        # Always keep last 6 exchanges in full detail
        recent_cutoff = 6
        recent = current_history[-recent_cutoff:] if len(current_history) > recent_cutoff else current_history
        older = current_history[:-recent_cutoff] if len(current_history) > recent_cutoff else []

        # Add recent in full
        for user_msg, bot_msg in reversed(recent):
            pair_tokens = estimate_tokens(str(user_msg)) + estimate_tokens(str(bot_msg))
            if total_tokens + pair_tokens > max_tokens - 5000:  # Leave buffer
                break
            optimized.insert(0, [user_msg, bot_msg])
            total_tokens += pair_tokens

        # If room remains, add older exchanges as summaries
        if older and total_tokens < max_tokens - 10000:
            # Create rolling summary of older context
            older_summary = self._summarize_older_exchanges(older)
            summary_tokens = estimate_tokens(older_summary)
            if total_tokens + summary_tokens < max_tokens - 5000:
                # Insert as system message at start
                optimized.insert(0, [None, f"[Previous context summary: {older_summary}]"])

        return optimized

    def _summarize_older_exchanges(self, exchanges):
        """Create brief summary of older exchanges."""
        topics = set()
        for user_msg, bot_msg in exchanges[-20:]:  # Last 20 older
            if user_msg:
                # Extract key nouns/topics
                words = re.findall(r'[A-Z][a-z]{3,}', str(user_msg))
                topics.update(words[:3])

        if topics:
            return f"Previous discussions covered: {', '.join(list(topics)[:10])}. {len(exchanges)} earlier exchanges available in history."
        return f"{len(exchanges)} earlier exchanges truncated from context."

# Global chat context manager
chat_context_manager = ChatContextManager()


# =========================================================
# --- API CALL ---
# =========================================================
def build_messages_with_uploads(user_input, history, use_rag=False):
    messages = []
    for prompt in active_prompts:
        if prompt.get("active", False):
            messages.append({"role": "system", "content": prompt.get("content", "")})

    # Apply smart compression to history
    compressed_history = chat_context_manager.build_optimized_history(history, user_input)

    for user_msg, bot_msg in compressed_history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})

    model_id = current_params.get("model", "")
    has_image_files = any(f["type"] == "image" for f in uploaded_files)
    using_vision_override = False
    if has_image_files and current_params.get("model_vision", "").strip():
        model_id = current_params["model_vision"].strip()
        using_vision_override = True

    modality_info = get_model_modality_info(model_id)
    sv = modality_info["supports_vision"]
    supports_vision = True if using_vision_override else (sv is not False)
    print(f"[vision] model={model_id}  override={using_vision_override}  sv={sv}  supports_vision={supports_vision}  images={has_image_files}")

    # OpenRouter requires: text FIRST, then images
    text_parts = []
    image_parts = []
    doc_parts = []

    for file_info in uploaded_files:
        if file_info["type"] == "image":
            if supports_vision:
                stored_mime = file_info.get("mime_type")
                if not stored_mime:
                    mime_map = {".png": "image/png", ".gif": "image/gif",
                                ".webp": "image/webp", ".jpg": "image/jpeg",
                                ".jpeg": "image/jpeg", ".bmp": "image/jpeg"}
                    stored_mime = mime_map.get(file_info["extension"], "image/jpeg")
                image_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{stored_mime};base64,{file_info['base64']}"}
                })
            else:
                doc_parts.append({"type": "text", "text": f"[Image attached: {file_info['name']}]"})
        elif file_info["type"] == "document":
            try:
                with open(file_info["path"], "r", encoding="utf-8", errors="ignore") as f:
                    doc_content = f.read()
                if use_rag and len(doc_content) > 2000:
                    doc_content = simple_rag_extract(doc_content, user_input)
                doc_parts.append({"type": "text", "text": f"[Document: {file_info['name']}]\n{doc_content}\n[End Document]"})
            except Exception as e:
                doc_parts.append({"type": "text", "text": f"[Error reading {file_info['name']}: {str(e)}]"})

    # Text prompt first, then images — per OpenRouter docs
    if user_input.strip():
        text_parts.append({"type": "text", "text": user_input})

    content_parts = text_parts + doc_parts + image_parts

    if image_parts:
        messages.append({"role": "user", "content": content_parts})
    else:
        text_only = "\n\n".join(p["text"] for p in content_parts if p["type"] == "text")
        messages.append({"role": "user", "content": text_only or user_input})

    return messages

def call_openrouter_api(messages, settings):
    global _inference_active
    api_key = settings.get("api_key", "").strip()
    if not api_key:
        yield "ERROR: API Key not set in Settings."
        return

    url = "https://openrouter.ai/api/v1/chat/completions"

    # Detect if any message contains image content
    has_images = any(
        isinstance(m.get("content"), list) and
        any(p.get("type") == "image_url" for p in m["content"])
        for m in messages
    )

    model_name = settings.get("model", "?")
    print(f"[api] model={model_name}  images={'yes' if has_images else 'no'}  msgs={len(messages)}")

    try:
        # Get the model's declared supported parameters (from OpenRouter models API)
        model_info = get_model_modality_info(model_name)
        supported = set(model_info.get("supported_parameters", []))
        # If supported_parameters is empty (unknown model), allow common ones
        allow_all = not supported

        def supports(param):
            return allow_all or param in supported

        payload = {
            "model": model_name,
            "messages": messages,
            "stream": not has_images,  # some providers reject stream=true with images
        }

        if supports("temperature"):
            payload["temperature"] = float(settings.get("temperature", 0.7))
        if supports("top_p"):
            payload["top_p"] = float(settings.get("top_p", 0.9))

        freq_p = float(settings.get("frequency_penalty", 0.0))
        pres_p = float(settings.get("presence_penalty", 0.0))
        if supports("frequency_penalty") and freq_p != 0.0:
            payload["frequency_penalty"] = freq_p
        if supports("presence_penalty") and pres_p != 0.0:
            payload["presence_penalty"] = pres_p

        response_tokens = int(settings.get("response_tokens", 0))
        if supports("max_tokens") and response_tokens > 0:
            payload["max_tokens"] = response_tokens

        # Extended params — only send if model declares support and value is non-default
        ext_params = {
            "top_k": (0, lambda v: int(v) != 0),
            "min_p": (0.0, lambda v: float(v) != 0.0),
            "top_a": (0.0, lambda v: float(v) != 0.0),
            "repetition_penalty": (1.0, lambda v: float(v) != 1.0),
            "seed": (-1, lambda v: int(v) != -1),
        }
        for k, (default, nondefault) in ext_params.items():
            v = settings.get(k)
            if v is not None and supports(k) and nondefault(v):
                payload[k] = v

        print(f"[api] supported_params={sorted(supported) if supported else 'unknown'}")

    except ValueError as e:
        yield f"ERROR: Invalid setting type. {e}"
        return

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/oic-client",
        "X-Title": "OiC Client"
    }
    TOKEN_TO_FILTER_HALF = "<|begin of sentence|>"
    TOKEN_TO_FILTER_FULL = "<｜begin of sentence｜>"

    try:
        # Debug: print payload structure (truncate base64 data so terminal stays readable)
        debug_payload = json.loads(json.dumps(payload))
        for msg in debug_payload.get("messages", []):
            if isinstance(msg.get("content"), list):
                for part in msg["content"]:
                    if part.get("type") == "image_url":
                        url_val = part.get("image_url", {}).get("url", "")
                        part["image_url"]["url"] = url_val[:60] + f"...[{len(url_val)} chars]"
        print(f"[payload] {json.dumps(debug_payload, indent=2)}")

        response = requests.post(url, headers=headers, json=payload,
                                 stream=not has_images, timeout=120)
        if response.status_code != 200:
            raw = response.text
            try:
                err_body = response.json()
                msg = err_body.get('error', {}).get('message', '') or str(err_body)
            except Exception:
                msg = raw
            print(f"[api] ERROR {response.status_code} raw: {raw}")
            hint = ""
            if response.status_code == 429:
                hint = "\n💡 Rate limit — wait a moment and retry, or switch models."
            elif response.status_code == 400:
                hint = "\n💡 Bad request — check terminal for full error details."
            yield f"❌ API error ({response.status_code}): {msg}{hint}"
            return

        # Non-streaming path for image requests
        if has_images:
            try:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                if TOKEN_TO_FILTER_FULL in content:
                    content = content.replace(TOKEN_TO_FILTER_FULL, "")
                if TOKEN_TO_FILTER_HALF in content:
                    content = content.replace(TOKEN_TO_FILTER_HALF, "")
                yield content
            except Exception as e:
                yield f"💥 Failed to parse vision response: {e}\nRaw: {response.text[:500]}"
            return

        # Streaming path for text-only requests
        for line_bytes in response.iter_lines():
            if is_stop_requested():
                yield "⛔ Inference stopped by user."
                _inference_active = False
                return

            if not line_bytes: 
                continue

            try:
                line = line_bytes.decode('utf-8')
            except UnicodeDecodeError: 
                continue

            if line.startswith("data:"):
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]": 
                    break
                try:
                    chunk = json.loads(data_str)
                except Exception: 
                    continue

                choices = chunk.get("choices", [])
                if not choices: 
                    continue

                choice0 = choices[0]
                delta = choice0.get("delta", {})
                content = delta.get("content")

                if content:
                    if TOKEN_TO_FILTER_FULL in content: 
                        content = content.replace(TOKEN_TO_FILTER_FULL, "")
                    if TOKEN_TO_FILTER_HALF in content: 
                        content = content.replace(TOKEN_TO_FILTER_HALF, "")
                    if content: 
                        yield content

                if choice0.get("finish_reason") is not None: 
                    break

    except requests.exceptions.RequestException as e:
        yield f"❌ Request error: {e}"
    except Exception as e:
        yield f"💥 Unexpected error: {e}"
    finally:
        _inference_active = False


# =========================================================
# --- CHAT FUNCTIONALITY ---
# =========================================================
def convert_to_messages_format(history):
    new_history = []
    for user_msg, bot_msg in history:
        if user_msg: 
            new_history.append({"role": "user", "content": user_msg})
        if bot_msg: 
            new_history.append({"role": "assistant", "content": bot_msg})
    return new_history

def chat_interface(user_input, history_state, use_rag=False, log_enabled=False):
    global _last_save_time, uploaded_files, log_as_txt_enabled, last_logged_model

    log_as_txt_enabled = log_enabled
    clear_stop_event()

    default_max_tokens = DEFAULT_PARAMS["max_tokens"]
    prompt_tokens = calculate_total_prompt_tokens()
    context_limit = int(current_params.get("max_tokens", default_max_tokens))
    current_model = current_params.get("model", DEFAULT_MODEL)

    if current_params.get("model") == "No Preset Loaded" or not current_params.get("api_key"):
        if not user_input.strip():
            yield convert_to_messages_format(history_state), history_state, gr.update(value="", interactive=True), f"Context Usage: {prompt_tokens} / {context_limit} tokens", ""
            return
        error_msg = "ERROR: No valid preset is loaded. Please create and select a preset in Settings."
        history_state.append([user_input, error_msg])
        yield convert_to_messages_format(history_state), history_state, gr.update(value="", interactive=True), f"Context Usage: {prompt_tokens} / {context_limit} tokens", ""
        return

    if not user_input.strip() and not uploaded_files:
        yield convert_to_messages_format(history_state), history_state, gr.update(value="", interactive=True), f"Context Usage: {prompt_tokens} / {context_limit} tokens", ""
        return

    display_input = user_input
    if uploaded_files:
        file_names = [f"📎 {f['name']}" for f in uploaded_files]
        display_input = f"{' '.join(file_names)}\n\n{user_input}" if user_input.strip() else ' '.join(file_names)

    history_state.append([display_input, None])
    save_current_session(history_state)
    _last_save_time = time.time()

    messages = build_messages_with_uploads(user_input, history_state[:-1], use_rag)
    total_used = sum(estimate_tokens(str(m.get("content", ""))) for m in messages)
    token_readout_text = f"Context Usage: {total_used} / {context_limit} tokens"

    # Determine effective model (vision override if images attached)
    effective_settings = current_params.copy()
    if uploaded_files and any(f["type"] == "image" for f in uploaded_files):
        vision_model = current_params.get("model_vision", "").strip()
        if vision_model:
            effective_settings["model"] = vision_model

    yield convert_to_messages_format(history_state), history_state, gr.update(value="", interactive=False), token_readout_text, ""

    full_reply = ""
    for chunk in call_openrouter_api(messages, effective_settings):
        is_error = isinstance(chunk, str) and (chunk.startswith("ERROR:") or chunk.startswith("❌") or chunk.startswith("💥"))

        if is_error:
            # Show error in chat so user can see it; restore input box; keep files for retry
            history_state[-1][1] = chunk
            save_current_session(history_state)
            if log_as_txt_enabled:
                log_assistant_output(chunk, current_model)
            yield convert_to_messages_format(history_state), history_state, gr.update(interactive=True, value=user_input), token_readout_text, ""
            return

        if isinstance(chunk, str) and chunk.startswith("⛔ Inference stopped"):
            full_reply += "\n\n*(Inference stopped)*"
            history_state[-1][1] = full_reply
            save_current_session(history_state)
            if log_as_txt_enabled:
                log_assistant_output(full_reply, current_model)
            yield convert_to_messages_format(history_state), history_state, gr.update(interactive=True), token_readout_text, ""
            return

        full_reply += chunk
        history_state[-1][1] = full_reply
        yield convert_to_messages_format(history_state), history_state, gr.update(interactive=False), token_readout_text, ""

    history_state[-1][1] = full_reply
    save_current_session(history_state)
    _last_save_time = time.time()

    if log_as_txt_enabled and full_reply:
        log_assistant_output(full_reply, current_model)

    # Save to full history log
    chat_context_manager.add_exchange(display_input, full_reply)

    uploaded_files.clear()
    yield convert_to_messages_format(history_state), history_state, gr.update(interactive=True), token_readout_text, ""

# =========================================================
# --- SESSION MANAGEMENT ---
# =========================================================
def save_current_session(history):
    global chat_history, last_logged_index, active_session_name
    if not history or not any(user_msg for user_msg, _ in history): 
        return

    sanitized = [[user_msg, bot_msg if bot_msg else "*(Incomplete Response)*"] for user_msg, bot_msg in history if user_msg]
    is_continuation = False
    last_entry = chat_history[-1] if chat_history else None

    if last_entry:
        if active_session_name and active_session_name == last_entry.get("name"):
            is_continuation = True
        else:
            existing_hist = last_entry.get("history", [])
            if len(existing_hist) <= len(sanitized) and existing_hist == sanitized[:len(existing_hist)]:
                active_session_name = last_entry.get("name")
                last_logged_index = len(existing_hist)
                is_continuation = True

    if is_continuation and last_entry:
        updated_history = last_entry.get("history", []).copy()
        if len(sanitized) == len(updated_history) and len(sanitized) > 0:
            updated_history[-1] = sanitized[-1]
        elif len(sanitized) > len(updated_history):
            start_idx = len(updated_history)
            updated_history.extend(sanitized[start_idx:])
        chat_history[-1] = {"name": last_entry.get("name"), "history": updated_history}
        save_json(HISTORY_FILE, chat_history)
        last_logged_index = len(sanitized)
    else:
        first_message = None
        for user_msg, _ in history:
            if user_msg:
                first_message = user_msg
                break

        if not first_message:
            first_message = "New Chat"

        name_base = textwrap.shorten(first_message, width=30, placeholder="...")
        proposed_name = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {name_base}"
        chat_history.append({"name": proposed_name, "history": sanitized})
        save_json(HISTORY_FILE, chat_history)
        active_session_name = proposed_name
        last_logged_index = len(sanitized)

def new_chat(history):
    global last_logged_index, active_session_name, uploaded_files
    if history: 
        save_current_session(history)
    last_logged_index = 0
    active_session_name = None
    uploaded_files = []
    prompt_tokens = calculate_total_prompt_tokens()
    context_limit = current_params.get("max_tokens", DEFAULT_PARAMS["max_tokens"])
    return [], [], gr.update(value=""), gr.update(value=f"Context Usage: {prompt_tokens} / {context_limit} tokens"), "", gr.update(value=None), gr.update(value="", visible=False)

def retry_last_message(history_state, use_rag=False, log_enabled=False):
    """Retry the last message. uploaded_files preserved from error path so images re-send."""
    prompt_tokens = calculate_total_prompt_tokens()
    context_limit = current_params.get("max_tokens", DEFAULT_PARAMS["max_tokens"])

    if not history_state:
        yield convert_to_messages_format([]), [], gr.update(value="", interactive=True), f"Context Usage: {prompt_tokens} / {context_limit} tokens", ""
        return

    last_turn = history_state[-1]
    last_user_msg = last_turn[0]

    if not last_user_msg:
        yield convert_to_messages_format(history_state), history_state, gr.update(value="", interactive=True), f"Context Usage: {prompt_tokens} / {context_limit} tokens", ""
        return

    # Strip the "📎 filename\n\n" display prefix to get the actual user text
    clean_msg = last_user_msg
    if clean_msg.startswith("📎"):
        parts = clean_msg.split("\n\n", 1)
        clean_msg = parts[1].strip() if len(parts) > 1 else ""

    # Remove the last exchange (error reply) and resend
    # uploaded_files still contains the image from the failed send
    new_history_state = history_state[:-1]
    for result in chat_interface(clean_msg, new_history_state, use_rag, log_enabled):
        yield result


# =========================================================
# --- HISTORY FUNCTIONS ---
# =========================================================
def get_history_names():
    return list(reversed([s["name"] for s in chat_history]))

def load_history_session(name):
    if name is None: 
        return []
    for session in chat_history:
        if session["name"] == name: 
            return convert_to_messages_format(session["history"])
    return []

def clear_history_log():
    global chat_history, last_logged_index, active_session_name
    message = "History log already clear."

    if os.path.exists(HISTORY_FILE):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = f"{HISTORY_FILE}.{timestamp}.ahv"
        try:
            shutil.copy2(HISTORY_FILE, backup_file)
            os.remove(HISTORY_FILE)
            message = f"History log backed up to `{backup_file}` and cleared."
        except Exception as e:
            message = f"ERROR backing up/clearing log: {e}"

    chat_history = []
    last_logged_index = 0
    active_session_name = None
    prompt_tokens = calculate_total_prompt_tokens()
    context_limit = current_params.get("max_tokens", DEFAULT_PARAMS["max_tokens"])

    return [], gr.update(value=None, choices=[]), message, gr.update(value=f"Context Usage: {prompt_tokens} / {context_limit} tokens")

def copy_all_session_text(name):
    if name is None: 
        return "No session selected."
    for session in chat_history:
        if session["name"] == name:
            text = f"--- Session: {name} ---\n\n"
            for user_msg, bot_msg in session["history"]:
                text += f"USER: {user_msg}\nASSISTANT: {bot_msg}\n\n"
            return text
    return "Session not found."

# =========================================================
# --- PROMPTS MANAGEMENT ---
# =========================================================
def get_prompt_names():
    global prompts_data
    prompts_data = load_json(PROMPTS_FILE, {"prompts": [DEFAULT_PROMPT], "groups": ["default"]})
    result = []
    for p in prompts_data.get("prompts", []):
        name = p.get("name", "Unnamed")
        if p.get("active", False):
            name = f"✓ {name}"
        result.append(name)
    return result

def get_prompt_groups():
    global prompts_data
    prompts_data = load_json(PROMPTS_FILE, {"prompts": [DEFAULT_PROMPT], "groups": ["default"]})
    return prompts_data.get("groups", ["default"])

def load_prompt_data(prompt_name):
    global prompts_data
    prompts_data = load_json(PROMPTS_FILE, {"prompts": [DEFAULT_PROMPT], "groups": ["default"]})
    for p in prompts_data.get("prompts", []):
        if p.get("name") == prompt_name:
            return p.get("content", ""), p.get("active", False), p.get("group", "default")
    return "", False, "default"

def save_prompt(name, content, active, group):
    global prompts_data, active_prompts
    if not name or not name.strip(): 
        return "Error: Prompt name required.", gr.update(), gr.update(), gr.update()

    prompts_list = prompts_data.get("prompts", [])
    groups_list = prompts_data.get("groups", ["default"])

    if group not in groups_list:
        groups_list.append(group)
        prompts_data["groups"] = groups_list

    found = False
    for p in prompts_list:
        if p.get("name") == name:
            p["content"] = content
            p["active"] = active
            p["group"] = group
            found = True
            break

    if not found:
        prompts_list.append({"name": name, "content": content, "active": active, "group": group})

    prompts_data["prompts"] = prompts_list
    save_json(PROMPTS_FILE, prompts_data)
    active_prompts = [p for p in prompts_list if p.get("active", False)] or [DEFAULT_PROMPT]
    prompt_tokens = calculate_total_prompt_tokens()
    context_limit = current_params.get("max_tokens", DEFAULT_PARAMS["max_tokens"])

    return f"Prompt '{name}' saved.", gr.update(choices=get_prompt_names(), value=name), gr.update(choices=get_prompt_groups(), value=group), f"Context Usage: {prompt_tokens} / {context_limit} tokens"

def delete_prompt(name):
    global prompts_data, active_prompts
    prompts_list = [p for p in prompts_data.get("prompts", []) if p.get("name") != name]
    prompts_data["prompts"] = prompts_list
    save_json(PROMPTS_FILE, prompts_data)
    active_prompts = [p for p in prompts_list if p.get("active", False)] or [DEFAULT_PROMPT]
    prompt_tokens = calculate_total_prompt_tokens()
    context_limit = current_params.get("max_tokens", DEFAULT_PARAMS["max_tokens"])

    return f"Prompt '{name}' deleted.", gr.update(choices=get_prompt_names(), value=None), f"Context Usage: {prompt_tokens} / {context_limit} tokens"

def toggle_prompt_active(prompt_name, active):
    global prompts_data, active_prompts
    for p in prompts_data.get("prompts", []):
        if p.get("name") == prompt_name:
            p["active"] = active
            break
    save_json(PROMPTS_FILE, prompts_data)
    active_prompts = [p for p in prompts_data.get("prompts", []) if p.get("active", False)]
    prompt_tokens = calculate_total_prompt_tokens()
    context_limit = current_params.get("max_tokens", DEFAULT_PARAMS["max_tokens"])

    return f"Context Usage: {prompt_tokens} / {context_limit} tokens"

def activate_prompt_group(group_name):
    global prompts_data, active_prompts

    if not group_name:
        return "No group selected.", gr.update(), f"Context Usage: {calculate_total_prompt_tokens()} / {current_params.get('max_tokens', DEFAULT_PARAMS['max_tokens'])} tokens"

    prompts_list = prompts_data.get("prompts", [])
    activated_count = 0

    for p in prompts_list:
        if p.get("group", "default") == group_name:
            p["active"] = True
            activated_count += 1
        else:
            p["active"] = False

    prompts_data["prompts"] = prompts_list
    save_json(PROMPTS_FILE, prompts_data)
    active_prompts = [p for p in prompts_list if p.get("active", False)]
    prompt_tokens = calculate_total_prompt_tokens()
    context_limit = current_params.get("max_tokens", DEFAULT_PARAMS["max_tokens"])

    return f"Activated {activated_count} prompt(s) from group '{group_name}'.", gr.update(choices=get_prompt_names()), f"Context Usage: {prompt_tokens} / {context_limit} tokens"

def deactivate_all_prompts():
    global prompts_data, active_prompts

    prompts_list = prompts_data.get("prompts", [])
    deactivated_count = 0

    for p in prompts_list:
        if p.get("active", False):
            p["active"] = False
            deactivated_count += 1

    prompts_data["prompts"] = prompts_list
    save_json(PROMPTS_FILE, prompts_data)
    active_prompts = []
    prompt_tokens = calculate_total_prompt_tokens()
    context_limit = current_params.get("max_tokens", DEFAULT_PARAMS["max_tokens"])

    return f"Deactivated all {deactivated_count} prompt(s).", gr.update(choices=get_prompt_names()), f"Context Usage: {prompt_tokens} / {context_limit} tokens"

def create_prompt_group(group_name):
    global prompts_data
    groups = prompts_data.get("groups", ["default"])
    if group_name and group_name not in groups:
        groups.append(group_name)
        prompts_data["groups"] = groups
        save_json(PROMPTS_FILE, prompts_data)
    return gr.update(choices=groups, value=group_name)


# =========================================================
# --- PARAMS MANAGEMENT ---
# =========================================================
def get_preset_names():
    return list(params_data.get("presets", {}).keys())

def load_params_preset(preset_name):
    global current_params, params_data, last_logged_model

    if preset_name not in params_data.get("presets", {}):
        return [gr.update() for _ in range(16)] + ["Preset not found."] + [gr.update() for _ in range(5)]

    current_params = params_data["presets"][preset_name].copy()
    params_data["active_preset"] = preset_name
    save_json(PARAMS_FILE, params_data)

    last_logged_model = None

    is_valid = bool(current_params.get("api_key"))
    return (
        current_params.get("api_key", ""),
        current_params.get("model", DEFAULT_MODEL),
        current_params.get("model_vision", ""),
        current_params.get("model_image_gen", ""),
        current_params.get("temperature", 0.7),
        current_params.get("max_tokens", 1024),
        current_params.get("response_tokens", 0),
        current_params.get("top_p", 0.9),
        current_params.get("frequency_penalty", 0.0),
        current_params.get("presence_penalty", 0.0),
        current_params.get("top_k", 0),
        current_params.get("min_p", 0.0),
        current_params.get("top_a", 0.0),
        current_params.get("repetition_penalty", 1.0),
        current_params.get("seed", -1),
        get_current_model_modality_display(),
        f"Loaded preset: {preset_name}",
        gr.update(interactive=is_valid),
        gr.update(interactive=is_valid),
        gr.update(interactive=is_valid),
        gr.update(interactive=is_valid),
        gr.update(interactive=is_valid),
    )

def save_params_preset(preset_name, api_key, model, model_vision, model_image_gen,
                       temperature, max_tokens, response_tokens,
                       top_p, freq_p, pres_p, top_k, min_p, top_a, rep_p, seed):
    global current_params, params_data, last_logged_model

    if not preset_name or not preset_name.strip(): 
        return "Error: Preset name required.", gr.update(), *[gr.update() for _ in range(5)]

    try:
        new_params = {
            "api_key": api_key, "model": model,
            "model_vision": model_vision.strip() if model_vision else "",
            "model_image_gen": model_image_gen.strip() if model_image_gen else "",
            "temperature": float(temperature),
            "max_tokens": int(max_tokens), "response_tokens": int(response_tokens),
            "top_p": float(top_p), "frequency_penalty": float(freq_p),
            "presence_penalty": float(pres_p), "top_k": int(top_k),
            "min_p": float(min_p), "top_a": float(top_a),
            "repetition_penalty": float(rep_p), "seed": int(seed),
        }
    except ValueError as e: 
        return f"Error: Invalid value type. {e}", gr.update(), *[gr.update() for _ in range(5)]

    params_data["presets"][preset_name] = new_params
    params_data["active_preset"] = preset_name
    save_json(PARAMS_FILE, params_data)
    current_params = new_params.copy()

    last_logged_model = None

    is_valid = bool(api_key)
    return (
        f"Saved preset: {preset_name}",
        gr.update(choices=get_preset_names(), value=preset_name),
        gr.update(interactive=is_valid), gr.update(interactive=is_valid),
        gr.update(interactive=is_valid), gr.update(interactive=is_valid),
        gr.update(interactive=is_valid),
    )

def create_new_params_preset(name):
    global params_data
    if not name or not name.strip():
        res = load_params_preset(params_data.get("active_preset", "Default"))
        return (*res, gr.update())

    name = name.strip()
    if name in params_data.get("presets", {}):
        res = load_params_preset(params_data.get("active_preset", "Default"))
        # res has: api_key,model,model_vision,model_image_gen,temp,...,seed,modality_display,msg,*chat_controls
        # Replace the message field (index -6, before 5 chat_controls)
        res_list = list(res)
        res_list[-6] = f"Error: '{name}' already exists."
        return (*res_list, gr.update())

    params_data["presets"][name] = DEFAULT_PARAMS.copy()
    save_json(PARAMS_FILE, params_data)
    res = load_params_preset(name)
    return (*res, gr.update(choices=get_preset_names(), value=name))

def delete_params_preset(name):
    global params_data
    if name in params_data.get("presets", {}):
        del params_data["presets"][name]
        remaining = list(params_data["presets"].keys())
        params_data["active_preset"] = remaining[0] if remaining else "Default"
        if not remaining: 
            params_data["presets"]["Default"] = DEFAULT_PARAMS.copy()
        save_json(PARAMS_FILE, params_data)

    return load_params_preset(params_data["active_preset"]) + (gr.update(choices=get_preset_names()),)

def update_model_from_input(model_value):
    global current_params, last_logged_model
    if model_value and model_value.strip():
        current_params["model"] = model_value.strip()
        last_logged_model = None
        return f"Model set to: {model_value.strip()}", get_current_model_modality_display()
    return "Model name cannot be empty", get_current_model_modality_display()

def update_vision_model(model_value):
    """Live-write vision model override to current_params on every field change."""
    global current_params
    current_params["model_vision"] = model_value.strip() if model_value else ""
    return get_current_model_modality_display()

def update_image_gen_model(model_value):
    """Live-write image-gen model override to current_params on every field change."""
    global current_params
    current_params["model_image_gen"] = model_value.strip() if model_value else ""
    return get_current_model_modality_display()

# =========================================================
# --- MODEL GROUPS MANAGEMENT ---
# =========================================================
def get_model_group_names():
    global model_groups_data
    model_groups_data = load_json(MODEL_GROUP_FILE, {})
    return list(model_groups_data.keys())

def load_model_group(group_name):
    global model_groups_data
    model_groups_data = load_json(MODEL_GROUP_FILE, {})
    if group_name in model_groups_data:
        g = model_groups_data[group_name]
        return g.get("primary", ""), g.get("vision", ""), g.get("image_gen", ""), f"Loaded group: {group_name}"
    return "", "", "", "Group not found."

def save_model_group(group_name, primary, vision, image_gen):
    global model_groups_data
    if not group_name or not group_name.strip():
        return "Error: Group name required.", gr.update()
    model_groups_data = load_json(MODEL_GROUP_FILE, {})
    model_groups_data[group_name.strip()] = {
        "primary": primary.strip(),
        "vision": vision.strip(),
        "image_gen": image_gen.strip()
    }
    save_json(MODEL_GROUP_FILE, model_groups_data)
    return f"Saved model group: {group_name}", gr.update(choices=get_model_group_names(), value=group_name.strip())

def delete_model_group(group_name):
    global model_groups_data
    model_groups_data = load_json(MODEL_GROUP_FILE, {})
    if group_name in model_groups_data:
        del model_groups_data[group_name]
        save_json(MODEL_GROUP_FILE, model_groups_data)
    remaining = get_model_group_names()
    return f"Deleted group: {group_name}", gr.update(choices=remaining, value=remaining[0] if remaining else None)

def activate_model_group(group_name):
    """Apply a model group to current_params: sets primary, vision override, image_gen override."""
    global current_params, model_groups_data, last_logged_model
    model_groups_data = load_json(MODEL_GROUP_FILE, {})
    if group_name not in model_groups_data:
        return "Group not found.", gr.update(), gr.update(), gr.update(), gr.update(), get_current_model_modality_display()
    g = model_groups_data[group_name]
    primary = g.get("primary", "").strip()
    vision = g.get("vision", "").strip()
    image_gen = g.get("image_gen", "").strip()
    if primary:
        current_params["model"] = primary
    current_params["model_vision"] = vision
    current_params["model_image_gen"] = image_gen
    last_logged_model = None
    # Save back to active preset
    active_preset = params_data.get("active_preset", "Default")
    if active_preset in params_data.get("presets", {}):
        params_data["presets"][active_preset].update({
            "model": current_params["model"],
            "model_vision": vision,
            "model_image_gen": image_gen
        })
        save_json(PARAMS_FILE, params_data)
    return (
        f"✅ Activated model group '{group_name}'",
        gr.update(value=current_params["model"]),
        gr.update(value=vision),
        gr.update(value=image_gen),
        gr.update(value=current_params["model"]),  # also update main model_display
        get_current_model_modality_display()
    )

# =========================================================
# --- THEME MANAGEMENT ---
# =========================================================
def apply_theme(theme_name):
    global current_theme
    current_theme = THEMES.get(theme_name, DEFAULT_THEME).copy()
    save_json(THEME_FILE, {"active_theme": theme_name})
    return f"Theme '{theme_name}' applied. (Refresh to see full effect)"

def generate_theme_css(theme):
    return f"""
    :root {{
        --primary-color: {theme['primary_color']};
        --secondary-color: {theme['secondary_color']};
        --background-color: {theme['background_color']};
        --surface-color: {theme['surface_color']};
        --text-color: {theme['text_color']};
        --border-color: {theme['border_color']};
        --accent-color: {theme['accent_color']};
    }}
    .gradio-container {{ background-color: var(--background-color) !important; color: var(--text_color) !important; }}
    .gradio-container .block {{ background-color: var(--surface-color) !important; border-color: var(--border-color) !important; }}
    .gradio-container button.primary {{ background-color: var(--primary-color) !important; }}
    .gradio-container button.secondary {{ background-color: var(--secondary-color) !important; }}
    .gradio-container .tabs > button.selected {{ border-bottom-color: var(--primary-color) !important; }}
    """

def get_custom_css():
    base_css = """
    footer { visibility: hidden !important; height: 0 !important; }
    .gradio-container button[aria-label="Settings"] { display: none !important; }
    .resizable-panel { resize: horizontal; overflow: auto; min-width: 300px; max-width: 80vw; }
    .chatbot-messages { scroll-behavior: smooth; }
    #token_readout { font-weight: bold; text-align: center; padding: 5px; margin: 5px; border-radius: 8px; }
    .file-attachment-bar { display: flex; align-items: center; gap: 8px; padding: 8px; border-radius: 8px; margin-bottom: 8px; }
    .model-list { max-height: 400px; overflow-y: auto; }
    .model-item { padding: 8px; cursor: pointer; border-radius: 4px; transition: background-color 0.2s; }
    .model-item:hover { background-color: rgba(128, 128, 128, 0.2); }
    .prompt-group-badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; margin-left: 8px; }
    .gradio-container .main { padding-left: 0 !important; padding-right: 0 !important; }
    .gradio-container .block { padding: 0 !important; }
    .gradio-container .tabitem { padding: 10px !important; }
    .gradio-container .gradio-chatbot { width: 100% !important; margin: 0 !important; }
    .resize-handle { width: 5px; cursor: col-resize; background: linear-gradient(to bottom, transparent, var(--border-color, #ccc), transparent); transition: background 0.2s; }
    .resize-handle:hover { background: linear-gradient(to bottom, transparent, var(--primary-color, #3b82f6), transparent); }
    .model-dropdown { min-width: 500px !important; }

    /* Auto-scroll improvements */
    .gradio-chatbot {
        scroll-behavior: smooth !important;
    }
    .gradio-chatbot > div {
        scroll-behavior: smooth !important;
    }
    /* Focus indicator for textarea */
    textarea:focus {
        outline: 2px solid var(--primary-color, #3b82f6) !important;
        outline-offset: 2px !important;
    }
    /* Compact file upload button */
    .compact-upload-btn {
        min-width: 32px !important;
        max-width: 32px !important;
        height: 32px !important;
        padding: 0 !important;
        font-size: 16px !important;
        line-height: 1 !important;
    }
    """
    return base_css + generate_theme_css(current_theme)


# =========================================================
# --- UI COMPONENTS ---
# =========================================================
CHATBOT_HEIGHT = "75vh"

def build_ui():
    prompt_tokens = calculate_total_prompt_tokens()
    context_limit = current_params.get("max_tokens", DEFAULT_PARAMS["max_tokens"])
    has_api_key = bool(current_params.get("api_key"))

    with gr.Blocks(title="OiC v2.5 - Clean") as app:
        session_history = gr.State(value=[])
        selected_model = gr.State(value=current_params.get("model", DEFAULT_MODEL))
        gr.Markdown(f"## 🤖 OiC Client v{VERSION}", visible=False)
        shutdown_message = gr.Markdown("", visible=False)

        with gr.Row(variant="compact", equal_height=False):
            quit_btn = gr.Button("❌", variant="stop", scale=0, min_width=40)
            with gr.Column(scale=100):
                with gr.Tabs() as tabs:
                    # --- CHAT TAB (Redesigned) ---
                    with gr.TabItem("💬"):
                        context_token_readout = gr.Textbox(
                            label="Context", 
                            value=f"{prompt_tokens} / {context_limit} tokens", 
                            interactive=False, 
                            elem_id="token_readout",
                            show_label=True
                        )

                        # Main chatbot - maximized space
                        chatbot = gr.Chatbot(
                            label="", 
                            height=CHATBOT_HEIGHT, 
                            elem_classes=["chatbot-messages"],
                            show_label=False
                        )

                        # File upload popup (hidden by default)
                        with gr.Row(visible=False) as file_upload_row:
                            file_upload = gr.File(
                                label="Select files", 
                                file_count="multiple",
                                interactive=True
                            )
                            close_upload_btn = gr.Button("✓ Done", variant="primary", size="sm")

                        # File status indicator (compact, inline)
                        file_status = gr.Markdown(
                            value="",
                            visible=False
                        )

                        # Input area - compact layout
                        with gr.Row(elem_id="user_input_wrapper", variant="compact"):
                            with gr.Column(scale=12):
                                user_input = gr.Textbox(
                                    placeholder="Type your message...", 
                                    label="", 
                                    show_label=False, 
                                    lines=2,
                                    interactive=has_api_key
                                )
                            with gr.Column(scale=1, min_width=40):
                                # Compact + button for file upload
                                upload_toggle_btn = gr.Button(
                                    "+", 
                                    variant="secondary",
                                    size="sm",
                                    elem_classes=["compact-upload-btn"]
                                )

                        with gr.Row(variant="compact"):
                            with gr.Column(scale=6):
                                with gr.Row(variant="compact"):
                                    use_rag = gr.Checkbox(label="RAG", value=False, scale=1)
                                    log_as_txt_checkbox = gr.Checkbox(label="Log TXT", value=False, scale=1)
                            with gr.Column(scale=6):
                                with gr.Row(variant="compact", equal_height=True):
                                    send_btn = gr.Button("➤ Send", variant="primary", scale=2, interactive=has_api_key, min_width=80)
                                    stop_btn = gr.Button("⏹", variant="stop", scale=1, interactive=has_api_key, min_width=50)
                                    retry_last_btn = gr.Button("↻", variant="secondary", scale=1, interactive=has_api_key, min_width=50)
                                    new_chat_btn = gr.Button("✕", variant="secondary", scale=1, interactive=has_api_key, min_width=50)

                        log_status = gr.Markdown("")

                        chat_controls = [user_input, send_btn, stop_btn, retry_last_btn, new_chat_btn]
                        chat_outputs = [chatbot, session_history, user_input, context_token_readout, log_status]

                        # Event handlers
                        send_btn.click(
                            chat_interface, 
                            inputs=[user_input, session_history, use_rag, log_as_txt_checkbox], 
                            outputs=chat_outputs
                        ).then(
                            fn=lambda: None,
                            inputs=None,
                            outputs=None,
                            js=focus_textbox_js()
                        )

                        user_input.submit(
                            chat_interface, 
                            inputs=[user_input, session_history, use_rag, log_as_txt_checkbox], 
                            outputs=chat_outputs
                        ).then(
                            fn=lambda: None,
                            inputs=None,
                            outputs=None,
                            js=focus_textbox_js()
                        )

                        stop_btn.click(set_stop_event, outputs=[log_status])

                        retry_last_btn.click(
                            retry_last_message, 
                            inputs=[session_history, use_rag, log_as_txt_checkbox], 
                            outputs=chat_outputs
                        )

                        new_chat_btn.click(
                            new_chat, 
                            inputs=[session_history], 
                            outputs=[chatbot, session_history, user_input, context_token_readout, log_status, file_upload, file_status]
                        )

                        # File upload toggle
                        upload_toggle_btn.click(
                            lambda: gr.Row(visible=True),
                            outputs=[file_upload_row]
                        )

                        close_upload_btn.click(
                            lambda: gr.Row(visible=False),
                            outputs=[file_upload_row]
                        )

                        # Handle file upload
                        file_upload.change(
                            handle_file_upload, 
                            inputs=[file_upload], 
                            outputs=[]
                        ).then(
                            lambda: gr.Markdown(visible=True, value=get_file_upload_status()),
                            outputs=[file_status]
                        ).then(
                            lambda: gr.Row(visible=False),
                            outputs=[file_upload_row]
                        )

                        # Retry event handler
                        chatbot.retry(
                            retry_last_message,
                            inputs=[session_history, use_rag, log_as_txt_checkbox],
                            outputs=chat_outputs
                        ).then(
                            fn=lambda: None,
                            inputs=None,
                            outputs=None,
                            js=focus_textbox_js()
                        )

                        log_as_txt_checkbox.change(
                            set_log_enabled,
                            inputs=[log_as_txt_checkbox],
                            outputs=[log_status]
                        )

                    # --- PROMPTS TAB ---
                    with gr.TabItem("📝 Prompts"):
                        gr.Markdown("### Prompt Management")

                        with gr.Row():
                            gr.Markdown("#### Quick Group Activation")
                        with gr.Row():
                            group_activate_dropdown = gr.Dropdown(
                                label="Select Group to Activate", 
                                choices=get_prompt_groups(), 
                                value=None, 
                                interactive=True,
                                scale=2
                            )
                            activate_group_btn = gr.Button("🎯 Activate Group", variant="primary", scale=1)
                            deactivate_all_btn = gr.Button("⭕ Deactivate All", variant="stop", scale=1)

                        with gr.Row():
                            new_group_name = gr.Textbox(label="New Group Name", placeholder="Create new group...", scale=2)
                            new_group_btn = gr.Button("➕ Create Group", variant="secondary", scale=1)

                        group_status = gr.Markdown("Click a group to activate all its prompts (deactivates others), or deactivate all prompts.")

                        gr.Markdown("---")

                        with gr.Row():
                            prompt_dropdown = gr.Dropdown(label="Select Prompt", choices=get_prompt_names(), value=None, interactive=True, scale=2)
                            prompt_active_checkbox = gr.Checkbox(label="Active", value=False, scale=1)
                            new_prompt_btn = gr.Button("✨ New", variant="secondary")
                            delete_prompt_btn = gr.Button("🗑️ Delete", variant="stop")
                        prompt_name_input = gr.Textbox(label="Prompt Name", value="")
                        prompt_content_input = gr.Textbox(label="Prompt Content", lines=5, value="")
                        prompt_group_dropdown = gr.Dropdown(label="Group", choices=get_prompt_groups(), value="default", interactive=True)
                        save_prompt_btn = gr.Button("💾 Save Prompt", variant="primary")
                        prompt_message = gr.Markdown("")

                        prompt_dropdown.change(
                            lambda name: (*load_prompt_data(name), name), 
                            inputs=[prompt_dropdown], 
                            outputs=[prompt_content_input, prompt_active_checkbox, prompt_group_dropdown, prompt_name_input]
                        )
                        save_prompt_btn.click(
                            save_prompt, 
                            inputs=[prompt_name_input, prompt_content_input, prompt_active_checkbox, prompt_group_dropdown], 
                            outputs=[prompt_message, prompt_dropdown, prompt_group_dropdown, context_token_readout]
                        )
                        delete_prompt_btn.click(
                            delete_prompt, 
                            inputs=[prompt_name_input], 
                            outputs=[prompt_message, prompt_dropdown, context_token_readout]
                        )
                        prompt_active_checkbox.change(
                            toggle_prompt_active, 
                            inputs=[prompt_name_input, prompt_active_checkbox], 
                            outputs=[context_token_readout]
                        )
                        new_prompt_btn.click(
                            lambda: (gr.update(value="New Prompt"), gr.update(value=""), gr.update(value=False)), 
                            outputs=[prompt_name_input, prompt_content_input, prompt_active_checkbox]
                        )
                        activate_group_btn.click(
                            activate_prompt_group,
                            inputs=[group_activate_dropdown],
                            outputs=[group_status, prompt_dropdown, context_token_readout]
                        )
                        deactivate_all_btn.click(
                            deactivate_all_prompts,
                            outputs=[group_status, prompt_dropdown, context_token_readout]
                        )
                        def create_and_refresh_group(name):
                            result = create_prompt_group(name)
                            return result, result
                        new_group_btn.click(
                            create_and_refresh_group,
                            inputs=[new_group_name],
                            outputs=[group_activate_dropdown, prompt_group_dropdown]
                        )

                    # --- PARAMS TAB ---
                    with gr.TabItem("⚙️ Params"):
                        gr.Markdown("### Model Parameters")
                        with gr.Row():
                            params_dropdown = gr.Dropdown(label="Select Preset", choices=get_preset_names(), value=params_data.get("active_preset", "Default"), interactive=True, scale=2)
                            new_preset_name = gr.Textbox(label="New Preset Name", scale=1)
                            new_params_btn = gr.Button("✨ New", variant="secondary", scale=1)
                            delete_params_btn = gr.Button("🗑️ Delete", variant="stop", scale=1)
                        params_message = gr.Markdown("")

                        # --- Current model modality info ---
                        current_model_modality = gr.Markdown(
                            value=get_current_model_modality_display(),
                            label="Active Model Capabilities",
                            elem_id="model_modality_display"
                        )

                        # --- Primary model row ---
                        with gr.Row():
                            model_display = gr.Textbox(
                                label="Primary Model", 
                                value=current_params.get("model", DEFAULT_MODEL), 
                                interactive=True, 
                                scale=3
                            )
                            update_model_btn = gr.Button("✓ Set", variant="secondary", scale=1)
                            browse_models_btn = gr.Button("🔍 Browse", variant="primary", scale=1)

                        # --- Per-modality override fields ---
                        with gr.Row():
                            model_vision_input = gr.Textbox(
                                label="👁 Vision Model (optional override)",
                                placeholder="e.g. meta-llama/llama-4-scout:free",
                                value=current_params.get("model_vision", ""),
                                interactive=True,
                                scale=3
                            )
                            browse_vision_btn = gr.Button("🔍", variant="secondary", scale=1, min_width=40)
                        with gr.Row():
                            model_image_gen_input = gr.Textbox(
                                label="🎨 Image-Gen Model (optional override)",
                                placeholder="e.g. black-forest-labs/FLUX-1-schnell",
                                value=current_params.get("model_image_gen", ""),
                                interactive=True,
                                scale=3
                            )
                            browse_imggen_btn = gr.Button("🔍", variant="secondary", scale=1, min_width=40)

                        # Track which slot the browser is targeting
                        browser_target = gr.State(value="primary")  # "primary" | "vision" | "image_gen"

                        # --- Model browser panel ---
                        with gr.Row(visible=False, elem_classes=["model-browser-panel"]) as model_browser_row:
                            with gr.Column(scale=3):
                                with gr.Row():
                                    model_search = gr.Textbox(
                                        label="Search Models", 
                                        placeholder="Enter keywords...",
                                        scale=2
                                    )
                                    model_modality_filter = gr.Radio(
                                        label="Modality",
                                        choices=["all", "text", "vision", "image_gen"],
                                        value="all",
                                        scale=2
                                    )
                                model_sort = gr.Radio(
                                    label="Sort By", 
                                    choices=["rate", "context", "name"], 
                                    value="rate",
                                    scale=1
                                )
                                model_list = gr.Dropdown(
                                    label="Available Models", 
                                    choices=[], 
                                    interactive=True,
                                    elem_classes=["model-dropdown"],
                                    scale=3
                                )
                                with gr.Row():
                                    select_model_btn = gr.Button("✓ Select Model", variant="primary", scale=1)
                                    close_browser_btn = gr.Button("✕ Close", variant="secondary", scale=1)

                        gr.Markdown("---")

                        # --- Model Groups section ---
                        gr.Markdown("#### 🗂 Model Groups")
                        gr.Markdown("*Save named sets of models per modality. Activating a group sets all model slots at once.*")
                        with gr.Row():
                            model_group_dropdown = gr.Dropdown(
                                label="Model Group",
                                choices=get_model_group_names(),
                                value=None,
                                interactive=True,
                                scale=2
                            )
                            activate_model_group_btn = gr.Button("🎯 Activate Group", variant="primary", scale=1)
                            delete_model_group_btn = gr.Button("🗑️ Delete", variant="stop", scale=1)
                        with gr.Row():
                            new_model_group_name = gr.Textbox(label="Group Name", placeholder="Name for this group...", scale=2)
                            save_model_group_btn = gr.Button("💾 Save Current as Group", variant="secondary", scale=2)
                        model_group_status = gr.Markdown("")

                        gr.Markdown("---")

                        api_key_input = gr.Textbox(label="OpenRouter API Key", value=current_params.get("api_key", ""), type="password")
                        with gr.Row():
                            temperature_input = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, step=0.01, value=current_params.get("temperature", 0.7))
                            max_tokens_input = gr.Slider(label="Context Window", minimum=1024, maximum=200000, step=1024, value=current_params.get("max_tokens", 1024))
                        with gr.Row():
                            response_tokens_input = gr.Slider(label="Response Tokens (0 = auto)", minimum=0, maximum=32000, step=128, value=current_params.get("response_tokens", 0))
                            top_p_input = gr.Slider(label="Top-P", minimum=0.0, maximum=1.0, step=0.01, value=current_params.get("top_p", 0.9))
                        with gr.Row():
                            freq_p_input = gr.Slider(label="Frequency Penalty", minimum=0.0, maximum=2.0, step=0.01, value=current_params.get("frequency_penalty", 0.0))
                            pres_p_input = gr.Slider(label="Presence Penalty", minimum=0.0, maximum=2.0, step=0.01, value=current_params.get("presence_penalty", 0.0))
                        with gr.Row():
                            top_k_input = gr.Slider(label="Top-K (0 = disabled)", minimum=0, maximum=200, step=1, value=current_params.get("top_k", 0))
                            min_p_input = gr.Slider(label="Min-P (0 = disabled)", minimum=0.0, maximum=1.0, step=0.01, value=current_params.get("min_p", 0.0))
                        with gr.Row():
                            top_a_input = gr.Slider(label="Top-A (0 = disabled)", minimum=0.0, maximum=1.0, step=0.01, value=current_params.get("top_a", 0.0))
                            rep_p_input = gr.Slider(label="Repetition Penalty", minimum=0.8, maximum=2.0, step=0.01, value=current_params.get("repetition_penalty", 1.0))
                        seed_input = gr.Number(label="Seed (-1 = random)", minimum=-1, value=current_params.get("seed", -1))
                        save_params_btn = gr.Button("💾 Save Preset", variant="primary")

                        # Collect all preset load/save outputs
                        preset_load_outputs = [
                            api_key_input, model_display, model_vision_input, model_image_gen_input,
                            temperature_input, max_tokens_input, response_tokens_input, top_p_input,
                            freq_p_input, pres_p_input, top_k_input, min_p_input, top_a_input,
                            rep_p_input, seed_input, current_model_modality, params_message,
                            *chat_controls
                        ]
                        preset_save_inputs = [
                            params_dropdown, api_key_input, model_display, model_vision_input, model_image_gen_input,
                            temperature_input, max_tokens_input, response_tokens_input, top_p_input,
                            freq_p_input, pres_p_input, top_k_input, min_p_input, top_a_input,
                            rep_p_input, seed_input
                        ]

                        params_dropdown.change(
                            load_params_preset, 
                            inputs=[params_dropdown], 
                            outputs=preset_load_outputs
                        )
                        save_params_btn.click(
                            save_params_preset, 
                            inputs=preset_save_inputs,
                            outputs=[params_message, params_dropdown, *chat_controls]
                        )
                        new_params_btn.click(
                            create_new_params_preset, 
                            inputs=[new_preset_name], 
                            outputs=[*preset_load_outputs, params_dropdown]
                        )
                        delete_params_btn.click(
                            delete_params_preset, 
                            inputs=[params_dropdown], 
                            outputs=[*preset_load_outputs, params_dropdown]
                        )
                        update_model_btn.click(
                            update_model_from_input,
                            inputs=[model_display],
                            outputs=[params_message, current_model_modality]
                        )
                        # Live-write vision/image_gen to current_params on every keystroke
                        model_vision_input.change(
                            update_vision_model,
                            inputs=[model_vision_input],
                            outputs=[current_model_modality]
                        )
                        model_image_gen_input.change(
                            update_image_gen_model,
                            inputs=[model_image_gen_input],
                            outputs=[current_model_modality]
                        )

                        # --- Model browser open/close/search/select ---
                        def open_browser_for(target):
                            return gr.Row(visible=True), gr.Dropdown(choices=search_models("", "rate", "all")), target

                        browse_models_btn.click(
                            lambda: (gr.Row(visible=True), gr.Dropdown(choices=search_models("", "rate", "all")), "primary"),
                            outputs=[model_browser_row, model_list, browser_target]
                        )
                        browse_vision_btn.click(
                            lambda: (gr.Row(visible=True), gr.Dropdown(choices=search_models("", "rate", "vision")), "vision"),
                            outputs=[model_browser_row, model_list, browser_target]
                        )
                        browse_imggen_btn.click(
                            lambda: (gr.Row(visible=True), gr.Dropdown(choices=search_models("", "rate", "image_gen")), "image_gen"),
                            outputs=[model_browser_row, model_list, browser_target]
                        )
                        close_browser_btn.click(
                            lambda: gr.Row(visible=False), 
                            outputs=[model_browser_row]
                        )
                        model_search.change(
                            lambda q, s, mf: gr.Dropdown(choices=search_models(q, s, mf)), 
                            inputs=[model_search, model_sort, model_modality_filter], 
                            outputs=[model_list]
                        )
                        model_sort.change(
                            lambda q, s, mf: gr.Dropdown(choices=search_models(q, s, mf)),
                            inputs=[model_search, model_sort, model_modality_filter],
                            outputs=[model_list]
                        )
                        model_modality_filter.change(
                            lambda q, s, mf: gr.Dropdown(choices=search_models(q, s, mf)),
                            inputs=[model_search, model_sort, model_modality_filter],
                            outputs=[model_list]
                        )

                        def apply_browser_selection(selected, target):
                            model_id = selected[1] if isinstance(selected, tuple) else selected
                            if not model_id:
                                return gr.update(), gr.update(), gr.update(), gr.Row(visible=False), gr.update()
                            # Write immediately to current_params — don't wait for Save Preset
                            if target == "vision":
                                current_params["model_vision"] = model_id
                            elif target == "image_gen":
                                current_params["model_image_gen"] = model_id
                            else:
                                current_params["model"] = model_id
                            info_md = get_current_model_modality_display()
                            if target == "vision":
                                return gr.update(), gr.update(value=model_id), gr.update(), gr.Row(visible=False), info_md
                            elif target == "image_gen":
                                return gr.update(), gr.update(), gr.update(value=model_id), gr.Row(visible=False), info_md
                            else:
                                return gr.update(value=model_id), gr.update(), gr.update(), gr.Row(visible=False), info_md

                        select_model_btn.click(
                            apply_browser_selection,
                            inputs=[model_list, browser_target],
                            outputs=[model_display, model_vision_input, model_image_gen_input, model_browser_row, current_model_modality]
                        )

                        # --- Model group handlers ---
                        model_group_dropdown.change(
                            lambda g: load_model_group(g),
                            inputs=[model_group_dropdown],
                            outputs=[model_display, model_vision_input, model_image_gen_input, model_group_status]
                        )
                        activate_model_group_btn.click(
                            activate_model_group,
                            inputs=[model_group_dropdown],
                            outputs=[model_group_status, model_display, model_vision_input, model_image_gen_input, model_display, current_model_modality]
                        )
                        save_model_group_btn.click(
                            save_model_group,
                            inputs=[new_model_group_name, model_display, model_vision_input, model_image_gen_input],
                            outputs=[model_group_status, model_group_dropdown]
                        )
                        delete_model_group_btn.click(
                            delete_model_group,
                            inputs=[model_group_dropdown],
                            outputs=[model_group_status, model_group_dropdown]
                        )

                    # --- HISTORY TAB ---
                    with gr.TabItem("📜 History"):
                        history_message = gr.Markdown("### Chat History")
                        with gr.Row():
                            history_dropdown = gr.Dropdown(label="Load Session", choices=get_history_names(), value=None, interactive=True, scale=2)
                            load_history_btn = gr.Button("⬇️ Load", variant="secondary")
                            clear_log_btn = gr.Button("❌ Clear All", variant="stop")
                        history_display = gr.Chatbot(label="Session Content", height=CHATBOT_HEIGHT)
                        copy_all_btn = gr.Button("📋 Copy All Text", variant="secondary")
                        copy_text_output = gr.Textbox(label="Copied Text", visible=False)
                        load_history_btn.click(load_history_session, inputs=[history_dropdown], outputs=[history_display])
                        history_dropdown.change(load_history_session, inputs=[history_dropdown], outputs=[history_display])
                        clear_log_btn.click(clear_history_log, outputs=[history_display, history_dropdown, history_message, context_token_readout])
                        copy_all_btn.click(copy_all_session_text, inputs=[history_dropdown], outputs=[history_message, copy_text_output])

                    # --- THEME TAB ---
                    with gr.TabItem("🎨 Theme"):
                        theme_dropdown = gr.Dropdown(label="Select Theme", choices=list(THEMES.keys()), value=current_theme.get("name", "Dark"), interactive=True)
                        theme_preview = gr.HTML(value="<div style='padding: 20px;'>Select a theme to preview</div>")
                        apply_theme_btn = gr.Button("✓ Apply Theme", variant="primary")
                        theme_message = gr.Markdown("")
                        def preview_theme(theme_name):
                            theme = THEMES.get(theme_name, DEFAULT_THEME)
                            return f"<div style='background: {theme['background_color']}; color: {theme['text_color']}; padding: 20px; border-radius: 8px;'><h3 style='color: {theme['primary_color']};'>Theme Preview: {theme_name}</h3><button style='background: {theme['primary_color']}; color: white; padding: 8px 16px; border: none; border-radius: 4px;'>Sample Button</button></div>"
                        theme_dropdown.change(preview_theme, inputs=[theme_dropdown], outputs=[theme_preview])
                        apply_theme_btn.click(apply_theme, inputs=[theme_dropdown], outputs=[theme_message])

        quit_btn.click(quit_app, outputs=[shutdown_message, quit_btn])

    return app

if __name__ == "__main__":
    PORT = random.randint(7860, 7960)
    local_url = f"http://127.0.0.1:{PORT}"
    print(f"\nOiC Client v{VERSION} starting on {local_url}")
    print("           ----------------------          ")
  
    app = build_ui()
    threading.Timer(2.0, lambda: webbrowser.open(local_url)).start()
    app.launch(server_name="0.0.0.0", server_port=PORT, ssl_verify=False, allowed_paths=["."], css=get_custom_css())