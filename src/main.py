import os
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"

import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import onnxruntime
import subprocess
import shutil


def bytes_to_gb(x):
    return round(int(x) / (1024**3), 2)


def get_nvidia_vram():
    if not shutil.which("nvidia-smi"):
        return None

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        gpus = []
        for line in output.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            name, mem = line.split(", ")
            gpus.append({
                "name": name,
                "dedicated_gb": round(int(mem) / 1024, 2)
            })
        return gpus
    except:
        return None


def get_windows_gpu_info():
    try:
        cmd = [
            "powershell",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json -Compress"
        ]
        output = subprocess.check_output(cmd, encoding="utf-8").strip()

        if not output:
            return []

        data = __import__("json").loads(output)
        if isinstance(data, dict):
            data = [data]

        gpus = []
        for item in data:
            name = str(item.get("Name", "")).strip()
            ram = item.get("AdapterRAM", None)
            if not name or ram in (None, "", 0):
                continue

            try:
                gpus.append({
                    "name": name,
                    "dedicated_gb": bytes_to_gb(ram)
                })
            except:
                pass

        return gpus
    except:
        return []


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class UpscaleAppLite:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Upscaler FP16")
        self.master.geometry("1000x850")

        self.tile_size_var = tk.IntVar(value=256)
        self.overlap_var = tk.IntVar(value=8)
        self.auto_mode = True

        self.pil_image = None
        self.upscaled_image = None
        self.upscale_session = None

        self.last_ui_update = 0
        self.cancel_flag = False

        self._setup_main_gui()
        self.load_upscale_model()
        self.detect_vram()

    def _setup_main_gui(self):
        # Status label for messages
        self.status_label = tk.Label(self.master, text="Initializing application...")
        self.status_label.pack(pady=6)

        # VRAM detection label
        self.vram_label = tk.Label(self.master, text="GPU: Detecting available VRAM...")
        self.vram_label.pack(pady=4)

        # Progress bar for tiles processing
        self.progress = ttk.Progressbar(self.master, orient="horizontal", length=700, mode="determinate")
        self.progress.pack(pady=5)

        # Frame to display image
        self.frame_image = tk.Frame(self.master, width=900, height=550, bg="gray")
        self.frame_image.pack(pady=6)

        self.image_label = tk.Label(self.frame_image, text="No image loaded", bg="gray")
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")

        # Buttons
        btn_frame = tk.Frame(self.master)
        btn_frame.pack(pady=10)

        self.btn_load = tk.Button(btn_frame, text="Load Image", command=self.load_image)
        self.btn_load.grid(row=0, column=0, padx=5)

        self.btn_upscale = tk.Button(btn_frame, text="Upscale (FP16)", command=self.start_upscale_thread, state=tk.DISABLED)
        self.btn_upscale.grid(row=0, column=1, padx=5)

        self.btn_save = tk.Button(btn_frame, text="Save Image", command=self.save_image, state=tk.DISABLED)
        self.btn_save.grid(row=0, column=2, padx=5)

        self.btn_cancel = tk.Button(btn_frame, text="Cancel", command=self.cancel_upscale, state=tk.DISABLED)
        self.btn_cancel.grid(row=0, column=3, padx=5)

        # Tiling settings
        ctrl_frame = tk.LabelFrame(self.master, text=" Tiling Settings ")
        ctrl_frame.pack(pady=10, padx=20, fill="x")

        tk.Label(ctrl_frame, text="Tile Size:").pack(side=tk.LEFT, padx=5)
        self.tile_spin = tk.Spinbox(
            ctrl_frame,
            from_=64,
            to=512,
            increment=64,
            textvariable=self.tile_size_var,
            width=5,
            state="disabled"
        )
        self.tile_spin.pack(side=tk.LEFT, padx=5)

        tk.Label(ctrl_frame, text="Overlap:").pack(side=tk.LEFT, padx=5)
        self.overlap_spin = tk.Spinbox(
            ctrl_frame,
            from_=8,
            to=64,
            increment=8,
            textvariable=self.overlap_var,
            width=5,
            state="disabled"
        )
        self.overlap_spin.pack(side=tk.LEFT, padx=5)

    def _apply_tiling_profile(self, tile, overlap, manual_enabled):
        self.tile_size_var.set(int(tile))
        self.overlap_var.set(int(overlap))
        self.auto_mode = not manual_enabled

        if manual_enabled:
            self.tile_spin.config(state="normal")
            self.overlap_spin.config(state="normal")
        else:
            self.tile_spin.config(state="disabled")
            self.overlap_spin.config(state="disabled")

    def detect_vram(self):
        def _detect():
            max_vram = 0
            gpu_text = "No GPU detected"

            nvidia = get_nvidia_vram()
            if nvidia:
                max_vram = max(g["dedicated_gb"] for g in nvidia)
                gpu_text = " | ".join([f"{g['name']} ({g['dedicated_gb']} GB)" for g in nvidia])
            else:
                gpus = get_windows_gpu_info()
                if gpus:
                    max_vram = max(g["dedicated_gb"] for g in gpus)
                    gpu_text = " | ".join([f"{g['name']} ({g['dedicated_gb']} GB)" for g in gpus])

            if max_vram <= 2:
                tile, overlap, manual = 128, 8, False
            elif max_vram <= 4:
                tile, overlap, manual = 256, 8, False
            elif max_vram <= 8:
                tile, overlap, manual = 256, 16, False
            else:
                tile, overlap, manual = 256, 16, True

            def _apply():
                self.vram_label.config(text=f"GPU: {gpu_text}")
                self._apply_tiling_profile(tile, overlap, manual)

            self.master.after(0, _apply)

        threading.Thread(target=_detect, daemon=True).start()

    def cancel_upscale(self):
        self.cancel_flag = True
        self.status_label.config(text="Cancelling operation...")

    def convert_model_fp16(self, input_path, output_path):
        try:
            import onnx
            from onnxconverter_common import float16

            self.master.after(0, lambda: self.status_label.config(
                text="⏳ Converting ONNX model to FP16 (first-time conversion)..."
            ))

            model = onnx.load(input_path)
            model_fp16 = float16.convert_float_to_float16(model, keep_io_types=False)
            onnx.save(model_fp16, output_path)

            return True
        except Exception as e:
            self.master.after(0, lambda: messagebox.showerror("Conversion Error", str(e)))
            return False

    def load_upscale_model(self):
        def _load():
            try:
                fp16_name = "4xNomosWebPhoto_esrgan_real_fp16.onnx"
                fp32_name = "4xNomosWebPhoto_esrgan_fp32_opset17.onnx"

                paths_fp16 = [
                    resource_path(fp16_name),
                    resource_path(f"models/{fp16_name}"),
                    os.path.join(os.getcwd(), fp16_name)
                ]

                paths_fp32 = [
                    resource_path(fp32_name),
                    resource_path(f"models/{fp32_name}"),
                    os.path.join(os.getcwd(), fp32_name)
                ]

                fp16_path = next((p for p in paths_fp16 if os.path.exists(p)), None)
                fp32_path = next((p for p in paths_fp32 if os.path.exists(p)), None)

                if fp16_path:
                    final_path = fp16_path

                elif fp32_path:
                    fp16_converted = fp32_path.replace(".onnx", "_real_fp16.onnx")

                    if not os.path.exists(fp16_converted):
                        ok = self.convert_model_fp16(fp32_path, fp16_converted)
                        if not ok:
                            return

                    final_path = fp16_converted

                else:
                    self.master.after(0, lambda: self.status_label.config(
                        text="❌ ONNX model not found"
                    ))
                    return

                self.master.after(0, lambda: self.status_label.config(text="⏳ Loading ONNX model..."))

                self.upscale_session = onnxruntime.InferenceSession(
                    final_path,
                    providers=['DmlExecutionProvider', 'CPUExecutionProvider']
                )

                self.master.after(0, lambda: self.status_label.config(text="✅ Model loaded and ready"))

                if self.pil_image:
                    self.master.after(0, lambda: self.btn_upscale.config(state=tk.NORMAL))

            except Exception as e:
                self.master.after(0, lambda: messagebox.showerror("Error", str(e)))

        threading.Thread(target=_load, daemon=True).start()

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if path:
            self.pil_image = Image.open(path).convert("RGB")
            self.show_preview(self.pil_image)
            if self.upscale_session:
                self.btn_upscale.config(state=tk.NORMAL)
            self.btn_save.config(state=tk.DISABLED)
            self.status_label.config(text=f"Loaded: {os.path.basename(path)}")

    def show_preview(self, img):
        w, h = img.size
        ratio = min(900 / w, 550 / h, 1)
        res = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(res)
        self.image_label.config(image=imgtk, text="")
        self.image_label.image = imgtk

    def start_upscale_thread(self):
        if self.upscale_session is None:
            return
        self.btn_upscale.config(state=tk.DISABLED)
        self.cancel_flag = False
        self.btn_cancel.config(state=tk.NORMAL)
        threading.Thread(target=self.process_upscale, daemon=True).start()

    def update_progress(self, idx, total):
        self.status_label.config(text=f"Processing tile {idx+1}/{total}")
        self.progress['value'] = (idx + 1) / total * 100

    def _is_oom_error(self, e):
        msg = str(e).lower()
        return (
            "out of memory" in msg
            or "not enough memory" in msg
            or "insufficient memory" in msg
            or "oom" in msg
        )

    def _process_once(self, tile, overlap):
        input_meta = self.upscale_session.get_inputs()[0]
        input_name = input_meta.name
        run = self.upscale_session.run

        img = self.pil_image
        w, h = img.size
        scale = 4

        out_img = Image.new("RGB", (w * scale, h * scale))
        tiles = [(x, y) for y in range(0, h, tile) for x in range(0, w, tile)]
        total = len(tiles)

        for idx, (x, y) in enumerate(tiles):
            if self.cancel_flag:
                self.master.after(0, self.on_cancel)
                return

            now = time.time()
            if now - self.last_ui_update > 0.03 or idx == total - 1:
                self.last_ui_update = now
                self.master.after(0, self.update_progress, idx, total)

            x1, y1 = max(0, x - overlap), max(0, y - overlap)
            x2, y2 = min(w, x + tile + overlap), min(h, y + tile + overlap)

            patch = img.crop((x1, y1, x2, y2))

            patch_np = np.asarray(patch, dtype=np.float16) / 255.0
            patch_np = patch_np.transpose(2, 0, 1)[None, ...]

            res = run(None, {input_name: patch_np})[0][0]

            res = np.clip(res.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
            patch_out = Image.fromarray(res)

            box_x1 = (x - x1) * scale
            box_y1 = (y - y1) * scale
            box_x2 = box_x1 + (min(x + tile, w) - x) * scale
            box_y2 = box_y1 + (min(y + tile, h) - y) * scale

            final_patch = patch_out.crop((box_x1, box_y1, box_x2, box_y2))
            out_img.paste(final_patch, (x * scale, y * scale))

        self.upscaled_image = out_img
        self.master.after(0, self.on_finish)

    def process_upscale(self):
        try:
            tile = int(self.tile_size_var.get())
            overlap = int(self.overlap_var.get())

            while True:
                try:
                    self._process_once(tile, overlap)
                    return
                except Exception as e:
                    if self.cancel_flag:
                        return

                    if self.auto_mode and self._is_oom_error(e):
                        if tile <= 64:
                            raise e

                        tile = max(64, tile - 64)
                        overlap = 8

                        def _apply_retry_values():
                            self.tile_size_var.set(tile)
                            self.overlap_var.set(overlap)
                            self.progress['value'] = 0
                            self.status_label.config(text=f"Out of memory, retrying with Tile {tile} | Overlap {overlap}")

                        self.master.after(0, _apply_retry_values)
                        continue

                    raise e

        except Exception as e:
            self.master.after(0, lambda: messagebox.showerror("Upscale Error", str(e)))
            self.master.after(0, lambda: self.btn_upscale.config(state=tk.NORMAL))
            self.master.after(0, lambda: self.btn_cancel.config(state=tk.DISABLED))

    def on_cancel(self):
        self.status_label.config(text="Operation cancelled")
        self.btn_upscale.config(state=tk.NORMAL)
        self.btn_cancel.config(state=tk.DISABLED)
        self.progress['value'] = 0

    def on_finish(self):
        self.progress['value'] = 100
        self.show_preview(self.upscaled_image)
        self.btn_save.config(state=tk.NORMAL)
        self.btn_upscale.config(state=tk.NORMAL)
        self.btn_cancel.config(state=tk.DISABLED)
        self.status_label.config(text="Upscaling completed")

    def save_image(self):
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")])
        if path:
            self.upscaled_image.save(path)


if __name__ == "__main__":
    root = tk.Tk()

    style = ttk.Style()
    style.theme_use('default')
    style.configure(
        "Horizontal.TProgressbar",
        troughcolor="#d9d9d9",
        background="#a6a6a6",
        lightcolor="#a6a6a6",
        darkcolor="#a6a6a6",
        bordercolor="#d9d9d9"
    )

    app = UpscaleAppLite(root)
    root.mainloop()
