import os
import sys
import gc
import json
import math
import time
import queue
import shutil
import threading
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk
import onnxruntime

# ============================================================
# Application constants
# ============================================================
APP_TITLE = "UpscaleLite-FP16"
APP_VERSION = "2.0"
MODEL_SCALE = 4
MIN_TILE = 64
MAX_TILE = 512
TILE_STEP = 64
DEFAULT_OVERLAP = 8
UI_UPDATE_THROTTLE_SEC = 0.03
PREVIEW_MAX_W = 960
PREVIEW_MAX_H = 600
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

MODEL_FP16_NAME = "4xNomosWebPhoto_esrgan_converted_fp16.onnx"
MODEL_FP32_NAME = "4xNomosWebPhoto_esrgan_fp32_opset17.onnx"
MODEL_OUT_SUFFIX = "_converted_fp16.onnx"

# ============================================================
# Utility functions
# ============================================================

def bytes_to_gb(x: Any) -> float:
    """Convert bytes to GB with 2 decimal points."""
    try:
        return round(int(x) / (1024 ** 3), 2)
    except Exception:
        return 0.0

def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default

def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))

def human_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 1:
        return "<1s"
    if seconds < 60:
        return f"{int(round(seconds))}s"
    minutes = int(seconds // 60)
    sec = int(round(seconds % 60))
    return f"{minutes}m {sec:02d}s"

def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS

def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS  # type: ignore[attr-defined]
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def get_working_directory() -> Path:
    return Path(os.getcwd()).resolve()

def load_json_text(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None

def try_which(executable: str) -> bool:
    return shutil.which(executable) is not None

def open_folder(path: Path) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass

# ============================================================
# GPU / VRAM detection
# ============================================================

def get_nvidia_vram() -> Optional[List[Dict[str, Any]]]:
    """Get NVIDIA GPU info using nvidia-smi if available."""
    if not try_which("nvidia-smi"):
        return None

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            encoding="utf-8",
            errors="replace",
        )
        gpus: List[Dict[str, Any]] = []
        for line in output.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            name, mem = parts[0], safe_int(parts[1], 0)
            if not name or mem <= 0:
                continue
            gpus.append({"name": name, "dedicated_gb": round(mem / 1024.0, 2), "kind": "nvidia"})
        return gpus
    except Exception:
        return None

def get_windows_gpu_info() -> List[Dict[str, Any]]:
    """Fallback GPU info using PowerShell/WMI."""
    try:
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json -Compress",
        ]
        output = subprocess.check_output(cmd, encoding="utf-8", errors="replace").strip()
        if not output:
            return []
        data = load_json_text(output)
        if data is None:
            return []
        if isinstance(data, dict):
            data = [data]

        gpus: List[Dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            name = str(item.get("Name", "")).strip()
            ram = item.get("AdapterRAM", None)
            if not name or ram in (None, "", 0):
                continue
            gpus.append({"name": name, "dedicated_gb": bytes_to_gb(ram), "kind": "windows"})
        return gpus
    except Exception:
        return []

def normalize_gpu_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize GPU entry with heuristics for shared memory."""
    name = str(entry.get("name", "Unknown GPU")).strip() or "Unknown GPU"
    gb = float(entry.get("dedicated_gb", 0.0) or 0.0)
    kind = str(entry.get("kind", "unknown"))
    lowered = name.lower()

    if "intel" in lowered or ("radeon" in lowered and "integrated" in lowered):
        gb = min(gb, 2.0)
    elif kind == "windows" and ("intel" in lowered or "uhd" in lowered or "iris" in lowered):
        gb = min(gb, 2.0)

    return {"name": name, "dedicated_gb": round(max(0.0, gb), 2), "kind": kind}

def detect_gpus() -> List[Dict[str, Any]]:
    """Detect GPUs with conservative fallback."""
    gpus: List[Dict[str, Any]] = []
    nvidia = get_nvidia_vram()
    if nvidia:
        gpus.extend(nvidia)
    else:
        gpus.extend(get_windows_gpu_info())
    return [normalize_gpu_entry(g) for g in gpus]

def choose_safe_profile(max_vram_gb: float) -> Tuple[int, int, bool]:
    """Map VRAM to tile size, overlap, manual mode."""
    if max_vram_gb <= 0:
        return 128, 8, False
    if max_vram_gb < 3:
        return 128, 8, False
    if max_vram_gb < 6:
        return 192, 8, False
    if max_vram_gb < 10:
        return 256, 16, False
    return 256, 16, True

def guess_output_path(input_path: Path, suffix: str = "_upscaled_x4") -> Path:
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")

# ============================================================
# Data structures
# ============================================================

@dataclass
class UpscaleJob:
    input_path: Path
    output_path: Path
    status: str = "queued"
    error: str = ""

# ============================================================
# Main Application
# ============================================================

class UpscaleAppLite:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title(f"{APP_TITLE} {APP_VERSION}")
        self.master.geometry("1180x920")
        self.master.minsize(1040, 820)

        self.work_dir = get_working_directory()
        self.output_dir = self.work_dir / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------
        # Runtime state
        # ---------------------------------------------------
        self.tile_size_var = tk.IntVar(value=256)
        self.overlap_var = tk.IntVar(value=8)
        self.auto_mode_var = tk.BooleanVar(value=True)
        self.use_fp16_var = tk.BooleanVar(value=True)
        self.preview_mode_var = tk.StringVar(value="original")
        self.batch_mode_var = tk.BooleanVar(value=False)
        self.keep_source_name_var = tk.BooleanVar(value=True)
        self.compare_view_var = tk.BooleanVar(value=False)
        self.status_detail_var = tk.StringVar(value="Ready")
        self.gpu_detail_var = tk.StringVar(value="Detecting GPUs...")
        self.model_detail_var = tk.StringVar(value="Model not loaded")
        self.progress_text_var = tk.StringVar(value="0%")
        self.eta_text_var = tk.StringVar(value="ETA: --")
        self.job_text_var = tk.StringVar(value="No active job")
        self.source_text_var = tk.StringVar(value="No image loaded")

        self.pil_image: Optional[Image.Image] = None
        self.upscaled_image: Optional[Image.Image] = None
        self.preview_original: Optional[Image.Image] = None
        self.preview_upscaled: Optional[Image.Image] = None
        self.current_image_path: Optional[Path] = None
        self.current_output_path: Optional[Path] = None
        self.batch_jobs: List[UpscaleJob] = []
        self.batch_running = False

        self.upscale_session: Optional[onnxruntime.InferenceSession] = None
        self.provider_name = "unknown"
        self.available_providers: List[str] = []

        self.cancel_flag = False
        self.model_loaded = False
        self.model_loading = False
        self.last_ui_update = 0.0
        self.tile_retry_count = 0
        self.job_start_ts = 0.0
        self.tile_timestamps: List[float] = []
        self.total_tiles = 0
        self.current_tile = 0

        self.ui_queue: "queue.Queue[Tuple[str, Tuple[Any, ...]]]" = queue.Queue()

        # ---------------------------------------------------
        # Setup UI
        # ---------------------------------------------------
        self._setup_styles()
        self._setup_main_gui()
        self._poll_ui_queue()

        # ---------------------------------------------------
        # Load model and detect GPU
        # ---------------------------------------------------
        self.load_upscale_model()
        self.detect_vram()

    # ---------------------------------------------------
    # --- UI setup methods
    # ---------------------------------------------------
    def _setup_styles(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("default")
        except Exception:
            pass
        try:
            style.configure(
                "Horizontal.TProgressbar",
                troughcolor="#d9d9d9",
                background="#a6a6a6",
                lightcolor="#a6a6a6",
                darkcolor="#a6a6a6",
                bordercolor="#d9d9d9",
            )
        except Exception:
            pass

    def _setup_main_gui(self) -> None:
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)
        root = tk.Frame(self.master)
        root.pack(fill="both", expand=True)

        # Top Info
        top_bar = tk.Frame(root)
        top_bar.pack(fill="x", padx=12, pady=8)
        tk.Label(top_bar, text=f"{APP_TITLE}", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        tk.Label(top_bar, text="Lightweight x4 image upscaler with DirectML, FP16, auto-VRAM profile, and OOM recovery.", fg="#555555").pack(anchor="w", pady=(2,0))

        # Status Row
        info_row = tk.Frame(root)
        info_row.pack(fill="x", padx=12, pady=(2,8))
        self.status_label = tk.Label(info_row, textvariable=self.status_detail_var, anchor="w")
        self.status_label.pack(side="left", fill="x", expand=True)
        tk.Label(info_row, textvariable=self.progress_text_var, width=8, anchor="e").pack(side="right", padx=(8,0))
        tk.Label(info_row, textvariable=self.eta_text_var, width=14, anchor="e").pack(side="right")

        # GPU Info
        gpu_row = tk.Frame(root)
        gpu_row.pack(fill="x", padx=12, pady=(0,4))
        tk.Label(gpu_row, textvariable=self.gpu_detail_var, anchor="w", justify="left", wraplength=1100).pack(fill="x")

        # Model Info
        model_row = tk.Frame(root)
        model_row.pack(fill="x", padx=12, pady=(0,8))
        tk.Label(model_row, textvariable=self.model_detail_var, anchor="w", fg="#444444").pack(fill="x")

        # Progress Bar
        self.progress = ttk.Progressbar(root, orient="horizontal", length=700, mode="determinate")
        self.progress.pack(fill="x", padx=12, pady=(0,10))

        # Main Area
        main_area = tk.Frame(root)
        main_area.pack(fill="both", expand=True, padx=12, pady=(0,10))
        left = tk.Frame(main_area)
        left.pack(side="left", fill="y")
        right = tk.Frame(main_area)
        right.pack(side="right", fill="both", expand=True)

        self._build_controls(left)
        self._build_preview(right)

        # Footer
        footer = tk.Frame(root)
        footer.pack(fill="x", padx=12, pady=(0,12))
        tk.Label(footer, textvariable=self.job_text_var, anchor="w", fg="#555555").pack(fill="x")

    def _build_controls(self, parent: tk.Widget) -> None:
        button_box = tk.LabelFrame(parent, text=" Actions ")
        button_box.pack(fill="x", pady=(0, 10))

        self.btn_load = tk.Button(button_box, text="Load Image", command=self.load_image, width=18)
        self.btn_load.grid(row=0, column=0, padx=6, pady=6, sticky="ew")

        self.btn_load_folder = tk.Button(button_box, text="Load Folder", command=self.load_folder, width=18)
        self.btn_load_folder.grid(row=1, column=0, padx=6, pady=6, sticky="ew")

        self.btn_upscale = tk.Button(button_box, text="Upscale Selected", command=self.start_upscale_thread, state=tk.DISABLED, width=18)
        self.btn_upscale.grid(row=2, column=0, padx=6, pady=6, sticky="ew")

        self.btn_batch = tk.Button(button_box, text="Batch Process", command=self.start_batch_thread, state=tk.DISABLED, width=18)
        self.btn_batch.grid(row=3, column=0, padx=6, pady=6, sticky="ew")

        self.btn_save = tk.Button(button_box, text="Save Output (single image)", command=self.save_image, state=tk.DISABLED, width=18)
        self.btn_save.grid(row=4, column=0, padx=6, pady=6, sticky="ew")

        self.btn_open_output = tk.Button(button_box, text="Open Output Folder (auto-saved)", command=lambda: open_folder(self.output_dir), width=18)
        self.btn_open_output.grid(row=5, column=0, padx=6, pady=6, sticky="ew")

        self.btn_cancel = tk.Button(button_box, text="Cancel", command=self.cancel_upscale, state=tk.DISABLED, width=18)
        self.btn_cancel.grid(row=6, column=0, padx=6, pady=6, sticky="ew")

        try:
            button_box.columnconfigure(0, weight=1)
        except Exception:
            pass

        settings_box = tk.LabelFrame(parent, text=" Tiling & Runtime ")
        settings_box.pack(fill="x", pady=(0, 10))

        row1 = tk.Frame(settings_box)
        row1.pack(fill="x", padx=8, pady=(8, 4))
        tk.Label(row1, text="Tile Size:").pack(side="left")
        self.tile_spin = tk.Spinbox(
            row1,
            from_=MIN_TILE,
            to=MAX_TILE,
            increment=TILE_STEP,
            textvariable=self.tile_size_var,
            width=6,
            state="disabled",
        )
        self.tile_spin.pack(side="left", padx=6)
        tk.Label(row1, text="Overlap:").pack(side="left", padx=(8, 0))
        self.overlap_spin = tk.Spinbox(
            row1,
            from_=8,
            to=64,
            increment=8,
            textvariable=self.overlap_var,
            width=6,
            state="disabled",
        )
        self.overlap_spin.pack(side="left", padx=6)

        row2 = tk.Frame(settings_box)
        row2.pack(fill="x", padx=8, pady=4)
        tk.Checkbutton(row2, text="Auto mode", variable=self.auto_mode_var, command=self.on_auto_mode_change).pack(anchor="w")
        tk.Checkbutton(row2, text="Compare view", variable=self.compare_view_var, command=self.refresh_preview).pack(anchor="w")
        tk.Checkbutton(row2, text="Keep original filename", variable=self.keep_source_name_var).pack(anchor="w")

        row3 = tk.Frame(settings_box)
        row3.pack(fill="x", padx=8, pady=(4, 8))
        tk.Label(row3, text="Preview:").pack(side="left")
        ttk.Combobox(
            row3,
            textvariable=self.preview_mode_var,
            values=["original", "side-by-side"],
            state="readonly",
            width=14,
        ).pack(side="left", padx=6)
        tk.Button(row3, text="Refresh", command=self.refresh_preview, width=10).pack(side="left", padx=4)

        self.batch_box = tk.LabelFrame(parent, text=" Batch Queue ")
        self.batch_box.pack(fill="both", expand=True)

        self.batch_list = tk.Listbox(self.batch_box, height=12)
        self.batch_list.pack(fill="both", expand=True, padx=8, pady=(8, 4))

        batch_btn_row = tk.Frame(self.batch_box)
        batch_btn_row.pack(fill="x", padx=8, pady=(0, 8))
        tk.Button(batch_btn_row, text="Clear Queue", command=self.clear_batch).pack(side="left")
        tk.Button(batch_btn_row, text="Remove Selected", command=self.remove_selected_batch).pack(side="left", padx=6)

    def _build_preview(self, parent: tk.Widget) -> None:
        preview_box = tk.LabelFrame(parent, text=" Preview ")
        preview_box.pack(fill="both", expand=True)

        self.preview_canvas = tk.Canvas(preview_box, bg="#777777", highlightthickness=0)
        self.preview_canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self.preview_canvas.bind("<Configure>", lambda _e: self.refresh_preview())

        self.preview_text_id = self.preview_canvas.create_text(
            10,
            10,
            anchor="nw",
            text="No image loaded",
            fill="white",
            font=("Segoe UI", 18, "bold"),
        )

        bottom = tk.Frame(preview_box)
        bottom.pack(fill="x", padx=8, pady=(0, 8))
        tk.Label(bottom, textvariable=self.source_text_var, anchor="w").pack(side="left", fill="x", expand=True)
        tk.Label(bottom, textvariable=self.job_text_var, anchor="e", fg="#666666").pack(side="right")

    # --------------------------------------------------------
    # Queue / UI dispatcher
    # --------------------------------------------------------
    def _post_ui(self, action: str, *args: Any) -> None:
        self.ui_queue.put((action, args))

    def _poll_ui_queue(self) -> None:
        try:
            while True:
                action, args = self.ui_queue.get_nowait()
                if action == "status":
                    self.status_detail_var.set(str(args[0]))
                elif action == "gpu":
                    self.gpu_detail_var.set(str(args[0]))
                elif action == "model":
                    self.model_detail_var.set(str(args[0]))
                elif action == "progress":
                    self._apply_progress_ui(*args)
                elif action == "preview":
                    self.refresh_preview()
                elif action == "upscale_done":
                    self.on_finish()
                elif action == "cancelled":
                    self.on_cancel()
                elif action == "error":
                    title, text = args
                    messagebox.showerror(str(title), str(text))
                elif action == "info":
                    title, text = args
                    messagebox.showinfo(str(title), str(text))
                elif action == "batch_refill":
                    self._rebuild_batch_list()
                elif action == "batch_state":
                    self._set_batch_running_ui(bool(args[0]))
                elif action == "enable_upscale":
                    self.btn_upscale.config(state=tk.NORMAL)
                elif action == "disable_upscale":
                    self.btn_upscale.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.master.after(60, self._poll_ui_queue)

    # --------------------------------------------------------
    # Buttons / mode handling
    # --------------------------------------------------------
    def on_auto_mode_change(self) -> None:
        enabled = not self.auto_mode_var.get()
        state = "normal" if enabled else "disabled"
        self.tile_spin.config(state=state)
        self.overlap_spin.config(state=state)

    def on_model_pref_change(self) -> None:
        # Reloading model can be added later; for now we keep the loaded model,
        # but we still expose the user's preference in the status text.
        pref = "FP16 preferred" if self.use_fp16_var.get() else "FP32 allowed"
        self.model_detail_var.set(f"Model preference updated: {pref}")

    def cancel_upscale(self) -> None:
        self.cancel_flag = True
        self._post_ui("status", "Cancelling current job...")
        self.btn_cancel.config(state=tk.DISABLED)

    def clear_batch(self) -> None:
        if self.batch_running:
            messagebox.showwarning("Batch running", "Stop the batch process before clearing the queue.")
            return
        self.batch_jobs.clear()
        self._rebuild_batch_list()

    def remove_selected_batch(self) -> None:
        if self.batch_running:
            messagebox.showwarning("Batch running", "Stop the batch process before removing items.")
            return
        indices = list(self.batch_list.curselection())
        if not indices:
            return
        for idx in reversed(indices):
            if 0 <= idx < len(self.batch_jobs):
                del self.batch_jobs[idx]
        self._rebuild_batch_list()

    # --------------------------------------------------------
    # Model conversion / loading
    # --------------------------------------------------------
    def convert_model_fp16(self, input_path: str, output_path: str) -> bool:
        try:
            import onnx
            from onnxconverter_common import float16
        except Exception as exc:
            self._post_ui("error", "Missing dependency", f"FP16 conversion dependencies are missing: {exc}")
            return False

        try:
            self._post_ui("status", "Converting ONNX model to FP16 for faster and lighter inference...")
            model = onnx.load(input_path)
            model_fp16 = float16.convert_float_to_float16(model, keep_io_types=False)
            onnx.save(model_fp16, output_path)
            return True
        except Exception as exc:
            self._post_ui("error", "Conversion Error", str(exc))
            return False

    def _resolve_model_paths(self) -> Tuple[Optional[str], Optional[str]]:
        fp16_name = MODEL_FP16_NAME
        fp32_name = MODEL_FP32_NAME

        paths_fp16 = [
            resource_path(fp16_name),
            resource_path(f"models/{fp16_name}"),
            str(get_working_directory() / fp16_name),
            str(get_working_directory() / "models" / fp16_name),
        ]
        paths_fp32 = [
            resource_path(fp32_name),
            resource_path(f"models/{fp32_name}"),
            str(get_working_directory() / fp32_name),
            str(get_working_directory() / "models" / fp32_name),
        ]


        fp16_path = next((p for p in paths_fp16 if os.path.exists(p)), None)
        fp32_path = next((p for p in paths_fp32 if os.path.exists(p)), None)


        if not fp16_path and fp32_path:
            fp16_converted = fp32_path.replace(".onnx", "_fp16.onnx")


        return fp16_path, fp32_path
    def _create_session(self, model_path: str) -> onnxruntime.InferenceSession:
        providers = []
        avail = onnxruntime.get_available_providers()
        self.available_providers = list(avail)

        # Prefer DirectML for Windows GPU acceleration, then CPU fallback.
        if "DmlExecutionProvider" in avail:
            providers.append("DmlExecutionProvider")
        if "CUDAExecutionProvider" in avail:
            providers.append("CUDAExecutionProvider")
        if "CPUExecutionProvider" in avail:
            providers.append("CPUExecutionProvider")

        if not providers:
            providers = ["CPUExecutionProvider"]

        return onnxruntime.InferenceSession(model_path, providers=providers)

    def load_upscale_model(self) -> None:
        if self.model_loading:
            return

        self.model_loading = True
        self._post_ui("status", "Loading ONNX model...")
        self._post_ui("model", "Loading model... please wait")

        def _load() -> None:
            try:
                fp16_path, fp32_path = self._resolve_model_paths()
                final_path: Optional[str] = None

                if self.use_fp16_var.get() and fp16_path:
                    if not os.path.exists(fp16_path) and fp32_path:
                        ok = self.convert_model_fp16(fp32_path, fp16_path)
                        if not ok:
                            self._post_ui("status", "FP16 conversion failed")
                            return
                    final_path = fp16_path
                elif fp32_path:
                    fp16_converted = fp32_path.replace(".onnx", MODEL_OUT_SUFFIX)
                    if not os.path.exists(fp16_converted):
                        ok = self.convert_model_fp16(fp32_path, fp16_converted)
                        if not ok:
                            self._post_ui("status", "FP16 conversion failed")
                            return
                    final_path = fp16_converted
                elif fp16_path:
                    final_path = fp16_path

                if not final_path or not os.path.exists(final_path):
                    self._post_ui("status", "ONNX model not found")
                    self._post_ui("model", "Model missing")
                    return

                # Load ONNX session
                session = self._create_session(final_path)
                self.upscale_session = session
                self.model_loaded = True

                providers = ", ".join(session.get_providers())
                self.provider_name = providers
                self._post_ui("model", f"Loaded: {Path(final_path).name} | Providers: {providers}")
                self._post_ui("status", "Model loaded and ready")


                if self.pil_image is not None:
                    self._post_ui("enable_upscale")
                if self.batch_jobs:
                    self.btn_batch.config(state=tk.NORMAL)

            except Exception as exc:
                self._post_ui("error", "Model Load Error", str(exc))
                self._post_ui("model", "Model load failed")
                self.model_loaded = False
            finally:
                self.model_loading = False


        threading.Thread(target=_load, daemon=True).start()

    # --------------------------------------------------------
    # VRAM / device detection
    # --------------------------------------------------------
    def detect_vram(self) -> None:
        def _detect() -> None:
            try:
                gpus = detect_gpus()
                if not gpus:
                    gpu_text = "GPU: No compatible GPU detected or hardware query failed."
                    tile, overlap, manual = 128, 8, False
                else:
                    max_vram = max(g["dedicated_gb"] for g in gpus)
                    gpu_text = "GPU: " + " | ".join(
                        f"{g['name']} ({g['dedicated_gb']} GB)" for g in gpus
                    )
                    tile, overlap, manual = choose_safe_profile(max_vram)

                self._post_ui("gpu", gpu_text)
                self._apply_tiling_profile(tile, overlap, manual)

            except Exception as exc:
                self._post_ui("gpu", f"GPU detection failed: {exc}")
                tile, overlap, manual = 128, 8, False
                self._apply_tiling_profile(tile, overlap, manual)

        threading.Thread(target=_detect, daemon=True).start()

    def _apply_tiling_profile(self, tile: int, overlap: int, manual_enabled: bool) -> None:
        def _apply() -> None:
            self.tile_size_var.set(int(tile))
            self.overlap_var.set(int(overlap))
            self.auto_mode_var.set(not manual_enabled)
            state = "normal" if manual_enabled else "disabled"
            self.tile_spin.config(state=state)
            self.overlap_spin.config(state=state)
            mode_text = "manual" if manual_enabled else "auto"
            self.status_detail_var.set(f"Runtime profile set: tile={tile}, overlap={overlap}, mode={mode_text}")
        self.master.after(0, _apply)

    # --------------------------------------------------------
    # Image / batch loading
    # --------------------------------------------------------
    def load_image(self) -> None:
        path_str = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Image files", "*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tif;*.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path_str:
            return
        path = Path(path_str)
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            return

        self.current_image_path = path
        self.pil_image = img
        self.upscaled_image = None
        self.preview_original = img.copy()
        self.preview_upscaled = None
        self.source_text_var.set(f"Source: {path.name}  |  {img.size[0]}×{img.size[1]}")
        self.job_text_var.set("Loaded single image")
        self.status_detail_var.set(f"Loaded: {path.name}")
        self.btn_save.config(state=tk.DISABLED)
        self._maybe_enable_upscale()
        self.refresh_preview()

    def load_folder(self) -> None:
        folder_str = filedialog.askdirectory(title="Select folder with images")
        if not folder_str:
            return
        folder = Path(folder_str)
        if not folder.exists():
            return

        paths = sorted([p for p in folder.iterdir() if p.is_file() and is_image_file(p)])
        if not paths:
            messagebox.showinfo("No images", "No supported image files were found in the folder.")
            return

        if self.batch_running:
            messagebox.showwarning("Batch running", "Batch is already running.")
            return

        self.batch_jobs.clear()
        for p in paths:
            out_name = p.name if self.keep_source_name_var.get() else guess_output_path(p).name
            output_path = self.output_dir / out_name
            self.batch_jobs.append(UpscaleJob(input_path=p, output_path=output_path))

        self._rebuild_batch_list()
        self.batch_mode_var.set(True)
        self.btn_batch.config(state=tk.NORMAL)
        self.status_detail_var.set(f"Loaded folder queue: {len(self.batch_jobs)} images")
        self.job_text_var.set(f"Batch queue ready: {len(self.batch_jobs)} files")

    def _rebuild_batch_list(self) -> None:
        self.batch_list.delete(0, tk.END)
        for idx, job in enumerate(self.batch_jobs, start=1):
            self.batch_list.insert(tk.END, f"{idx:03d}. {job.input_path.name} -> {job.output_path.name}")
        if self.batch_jobs:
            self.btn_batch.config(state=tk.NORMAL)
        else:
            self.btn_batch.config(state=tk.DISABLED)

    def _maybe_enable_upscale(self) -> None:
        if self.pil_image is not None and self.upscale_session is not None:
            self.btn_upscale.config(state=tk.NORMAL)
            self.btn_batch.config(state=tk.NORMAL if self.batch_jobs else tk.DISABLED)

    # --------------------------------------------------------
    # Preview helpers
    # --------------------------------------------------------
    def _fit_image(self, img: Image.Image, max_w: int, max_h: int) -> Image.Image:
        w, h = img.size
        ratio = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
        new_w = max(1, int(round(w * ratio)))
        new_h = max(1, int(round(h * ratio)))
        return img.resize((new_w, new_h), Image.LANCZOS)

    def _draw_preview_image(self, img: Image.Image) -> None:
        canvas = self.preview_canvas
        canvas.delete("preview_image")
        canvas.delete("preview_text")

        c_w = max(10, canvas.winfo_width())
        c_h = max(10, canvas.winfo_height())
        rendered = self._fit_image(img, c_w - 20, c_h - 20)
        imgtk = ImageTk.PhotoImage(rendered)
        canvas.image = imgtk  # type: ignore[attr-defined]
        canvas.create_image(c_w // 2, c_h // 2, image=imgtk, anchor="center", tags=("preview_image",))

    def refresh_preview(self) -> None:
        if self.pil_image is None and self.upscaled_image is None:
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(
                20,
                20,
                anchor="nw",
                text="No image loaded",
                fill="white",
                font=("Segoe UI", 18, "bold"),
                tags=("preview_text",),
            )
            return

        show_upscaled = self.compare_view_var.get() and self.upscaled_image is not None
        preview_mode = self.preview_mode_var.get()

        if preview_mode == "side-by-side" and self.pil_image is not None:
            left_img = self._fit_image(self.pil_image, (self.preview_canvas.winfo_width() // 2) - 20, self.preview_canvas.winfo_height() - 20)
            right_src = self.upscaled_image if show_upscaled and self.upscaled_image is not None else self.pil_image
            right_img = self._fit_image(right_src, (self.preview_canvas.winfo_width() // 2) - 20, self.preview_canvas.winfo_height() - 20)

            composite = Image.new(
                "RGB",
                (left_img.width + right_img.width + 16, max(left_img.height, right_img.height)),
                (80, 80, 80),
            )
            composite.paste(left_img, (0, 0))
            composite.paste(right_img, (left_img.width + 16, 0))
            self._draw_preview_image(composite)
            return

        if show_upscaled and self.upscaled_image is not None:
            self._draw_preview_image(self.upscaled_image)
        elif self.pil_image is not None:
            self._draw_preview_image(self.pil_image)

    # --------------------------------------------------------
    # Processing orchestration
    # --------------------------------------------------------
    def start_upscale_thread(self) -> None:
        if self.upscale_session is None or self.pil_image is None:
            return
        if self.batch_running:
            messagebox.showwarning("Batch running", "A batch job is already running.")
            return

        self.cancel_flag = False
        self.btn_upscale.config(state=tk.DISABLED)
        self.btn_cancel.config(state=tk.NORMAL)
        self.job_start_ts = time.time()
        self.tile_retry_count = 0
        self.tile_timestamps = []
        threading.Thread(target=self.process_upscale_single, daemon=True).start()

    def start_batch_thread(self) -> None:
        if self.upscale_session is None:
            return
        if not self.batch_jobs:
            messagebox.showinfo("Batch queue", "No batch items loaded.")
            return
        if self.batch_running:
            messagebox.showwarning("Batch running", "Batch is already running.")
            return

        self.cancel_flag = False
        self.batch_running = True
        self._post_ui("batch_state", True)
        self.btn_cancel.config(state=tk.NORMAL)
        self.job_start_ts = time.time()
        threading.Thread(target=self.process_batch_queue, daemon=True).start()

    def process_upscale_single(self) -> None:
        try:
            if self.pil_image is None or self.upscale_session is None:
                return
            tile = clamp(safe_int(self.tile_size_var.get(), 256), MIN_TILE, MAX_TILE)
            overlap = clamp(safe_int(self.overlap_var.get(), 8), 0, 64)
            result = self._process_image_with_retry(self.pil_image, tile, overlap)
            self.upscaled_image = result
            self.preview_upscaled = result.copy()
            self.current_output_path = None
            self._post_ui("upscale_done")
        except Exception as exc:
            self._post_ui("error", "Upscale Error", str(exc))
            self._post_ui("status", "Upscale failed")
            self.master.after(0, lambda: self.btn_upscale.config(state=tk.NORMAL))
            self.master.after(0, lambda: self.btn_cancel.config(state=tk.DISABLED))

    def process_batch_queue(self) -> None:
        try:
            for job in self.batch_jobs:
                if self.cancel_flag:
                    break

                try:
                    self._post_ui("status", f"Loading {job.input_path.name}...")
                    img = Image.open(job.input_path).convert("RGB")
                    self.current_image_path = job.input_path
                    self.pil_image = img
                    tile = clamp(safe_int(self.tile_size_var.get(), 256), MIN_TILE, MAX_TILE)
                    overlap = clamp(safe_int(self.overlap_var.get(), 8), 0, 64)
                    out_img = self._process_image_with_retry(img, tile, overlap)
                    self.upscaled_image = out_img
                    job.output_path.parent.mkdir(parents=True, exist_ok=True)
                    out_img.save(job.output_path)
                    job.status = "done"
                    self.current_output_path = job.output_path
                    self._post_ui("status", f"Saved: {job.output_path.name}")
                except Exception as exc:
                    job.status = "error"
                    job.error = str(exc)
                    self._post_ui("status", f"Failed: {job.input_path.name}")
                    self._post_ui("error", "Batch item failed", f"{job.input_path.name}\n\n{exc}")
                    # Continue to next file unless canceled.
                    continue

            self._post_ui("status", "Batch processing finished")
        finally:
            self.batch_running = False
            self._post_ui("batch_state", False)
            self.cancel_flag = False
            self._post_ui("preview")
            self._post_ui("enable_upscale")
            self.master.after(0, lambda: self.btn_cancel.config(state=tk.DISABLED))

    # --------------------------------------------------------
    # Low-level ONNX image processing
    # --------------------------------------------------------
    def _is_oom_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        keywords = [
            "out of memory",
            "not enough memory",
            "insufficient memory",
            "oom",
            "allocate memory",
            "allocation failed",
            "failed to allocate",
            "e_outofmemory",
            "device removed",
            "dxgi_error_device_removed",
        ]
        return any(k in msg for k in keywords)

    def _estimate_eta(self, current_idx: int, total: int) -> str:
        if current_idx <= 0 or not self.tile_timestamps:
            return "ETA: --"
        elapsed = time.time() - self.job_start_ts
        progress = current_idx / max(total, 1)
        if progress <= 0:
            return "ETA: --"
        estimated_total = elapsed / progress
        remain = max(0.0, estimated_total - elapsed)
        return f"ETA: {human_seconds(remain)}"

    def _apply_progress_ui(self, idx: int, total: int, elapsed_hint: Optional[float] = None) -> None:
        self.current_tile = idx
        self.total_tiles = total
        progress = (idx + 1) / max(total, 1) * 100.0
        self.progress["value"] = progress
        self.progress_text_var.set(f"{progress:.0f}%")
        eta = self._estimate_eta(idx + 1, total)
        self.eta_text_var.set(eta)
        self.status_detail_var.set(f"Processing tile {idx + 1}/{total}")
        if elapsed_hint is not None:
            self.job_text_var.set(f"Elapsed: {human_seconds(elapsed_hint)} | Tile {idx + 1}/{total}")

    def _run_inference_patch(self, patch: Image.Image) -> np.ndarray:
        if self.upscale_session is None:
            raise RuntimeError("Model session is not ready.")

        input_meta = self.upscale_session.get_inputs()[0]
        input_name = input_meta.name

        patch_np = np.asarray(patch, dtype=np.float32) / 255.0
        patch_np = patch_np.transpose(2, 0, 1)[None, ...]
        if self.use_fp16_var.get():
            patch_np = patch_np.astype(np.float16)

        outputs = self.upscale_session.run(None, {input_name: patch_np})
        if not outputs:
            raise RuntimeError("Model returned no outputs.")

        res = outputs[0]
        if isinstance(res, list):
            res = res[0]
        if isinstance(res, tuple):
            res = res[0]
        if res.ndim == 4:
            res = res[0]
        if res.ndim != 3:
            raise RuntimeError(f"Unexpected model output shape: {getattr(res, 'shape', None)}")

        return res

    def _process_once(self, img: Image.Image, tile: int, overlap: int) -> Image.Image:
        w, h = img.size
        scale = MODEL_SCALE

        out_img = Image.new("RGB", (w * scale, h * scale))
        tiles = [(x, y) for y in range(0, h, tile) for x in range(0, w, tile)]
        total = len(tiles)
        self.total_tiles = total
        self.tile_timestamps = []

        for idx, (x, y) in enumerate(tiles):
            if self.cancel_flag:
                raise KeyboardInterrupt("Cancelled by user")

            now = time.time()
            if (now - self.last_ui_update) > UI_UPDATE_THROTTLE_SEC or idx == total - 1:
                self.last_ui_update = now
                self.tile_timestamps.append(now)
                self.master.after(0, self._apply_progress_ui, idx, total, now - self.job_start_ts)

            x1 = max(0, x - overlap)
            y1 = max(0, y - overlap)
            x2 = min(w, x + tile + overlap)
            y2 = min(h, y + tile + overlap)

            patch = img.crop((x1, y1, x2, y2))
            res = self._run_inference_patch(patch)

            res = np.clip(res.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
            patch_out = Image.fromarray(res)

            box_x1 = (x - x1) * scale
            box_y1 = (y - y1) * scale
            box_x2 = box_x1 + (min(x + tile, w) - x) * scale
            box_y2 = box_y1 + (min(y + tile, h) - y) * scale

            final_patch = patch_out.crop((box_x1, box_y1, box_x2, box_y2))
            out_img.paste(final_patch, (x * scale, y * scale))

        return out_img

    def _process_image_with_retry(self, img: Image.Image, tile: int, overlap: int) -> Image.Image:
        current_tile = clamp(tile, MIN_TILE, MAX_TILE)
        current_overlap = clamp(overlap, 0, 64)
        attempt = 0

        while True:
            if self.cancel_flag:
                raise KeyboardInterrupt("Cancelled by user")

            try:
                attempt += 1
                self.tile_retry_count = attempt - 1
                self._post_ui(
                    "status",
                    f"Processing with tile={current_tile}, overlap={current_overlap} (attempt {attempt})",
                )
                gc.collect()
                return self._process_once(img, current_tile, current_overlap)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                if self.cancel_flag:
                    raise KeyboardInterrupt("Cancelled by user")

                if self.auto_mode_var.get() and self._is_oom_error(exc):
                    if current_tile <= MIN_TILE:
                        raise RuntimeError(
                            "Not enough VRAM even at the minimum tile size. Try closing other GPU apps or use a smaller image.") from exc

                    # Conservative retry strategy: jump down one step, reset overlap.
                    new_tile = max(MIN_TILE, current_tile - TILE_STEP)
                    if new_tile == current_tile:
                        new_tile = max(MIN_TILE, current_tile // 2)
                    current_tile = new_tile
                    current_overlap = DEFAULT_OVERLAP
                    self.tile_size_var.set(current_tile)
                    self.overlap_var.set(current_overlap)
                    self.last_ui_update = 0.0
                    self.progress["value"] = 0
                    self.progress_text_var.set("0%")
                    self.eta_text_var.set("ETA: --")
                    self.status_detail_var.set(
                        f"Out of memory. Retrying with tile={current_tile}, overlap={current_overlap}..."
                    )
                    continue

                raise

    # --------------------------------------------------------
    # Finish / cancel / save
    # --------------------------------------------------------
    def on_cancel(self) -> None:
        self.status_detail_var.set("Operation cancelled")
        self.job_text_var.set("No active job")
        self.progress["value"] = 0
        self.progress_text_var.set("0%")
        self.eta_text_var.set("ETA: --")
        self.btn_cancel.config(state=tk.DISABLED)
        self.btn_upscale.config(state=tk.NORMAL if self.pil_image is not None and self.model_loaded else tk.DISABLED)

    def on_finish(self) -> None:
        self.progress["value"] = 100
        self.progress_text_var.set("100%")
        self.eta_text_var.set("ETA: 0s")
        self.status_detail_var.set("Upscaling completed")
        self.job_text_var.set("Ready to save output")
        self.btn_save.config(state=tk.NORMAL)
        self.btn_upscale.config(state=tk.NORMAL if self.pil_image is not None and self.model_loaded else tk.DISABLED)
        self.btn_cancel.config(state=tk.DISABLED)
        self.refresh_preview()

    def save_image(self) -> None:
        if self.upscaled_image is None:
            messagebox.showinfo("Save image", "No upscaled image to save.")
            return

        default_name = "upscaled_x4.png"
        if self.current_image_path is not None and self.keep_source_name_var.get():
            default_name = f"{self.current_image_path.stem}_x4.png"

        path = filedialog.asksaveasfilename(
            title="Save image",
            initialdir=str(self.output_dir),
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPG", "*.jpg;*.jpeg"),
                ("WEBP", "*.webp"),
                ("BMP", "*.bmp"),
                ("TIFF", "*.tif;*.tiff"),
            ],
        )
        if not path:
            return

        try:
            out_path = Path(path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self.upscaled_image.save(out_path)
            self.current_output_path = out_path
            self.status_detail_var.set(f"Saved: {out_path.name}")
            self.job_text_var.set(f"Saved to {out_path}")
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------
    def _set_batch_running_ui(self, running: bool) -> None:
        self.batch_running = running
        if running:
            self.btn_batch.config(state=tk.DISABLED)
            self.btn_upscale.config(state=tk.DISABLED)
            self.btn_load.config(state=tk.DISABLED)
            self.btn_load_folder.config(state=tk.DISABLED)
        else:
            self.btn_load.config(state=tk.NORMAL)
            self.btn_load_folder.config(state=tk.NORMAL)
            self._maybe_enable_upscale()
            if self.batch_jobs:
                self.btn_batch.config(state=tk.NORMAL)

    def on_close(self) -> None:
        if self.batch_running or not self.cancel_flag:
            # Give a clear chance to abort active work.
            if messagebox.askokcancel("Exit", "Close the application? Active processing will be stopped."):
                self.cancel_flag = True
                self.master.after(150, self.master.destroy)
            return
        self.master.destroy()

# ============================================================
# Main entry point
# ============================================================

def main() -> None:
    root = tk.Tk()
    app = UpscaleAppLite(root)
    root.mainloop()

if __name__ == "__main__":
    main()
