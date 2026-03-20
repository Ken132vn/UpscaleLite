# UpscaleLite-FP16 (x4) with GUI

High-performance image upscaler optimized for low-to-mid end GPUs using ONNX Runtime and DirectML.

## Features

  - **Built-in Native GUI:** Simple and lightweight interface powered by **Tkinter**. Provides a seamless "point-and-click" experience without the overhead of heavy frameworks.
  - **Universal Hardware Acceleration:** Supports Nvidia, AMD, and Intel GPUs via DirectML execution provider for high-speed cross-vendor inference.
  - **Advanced VRAM Efficiency:** Features an automated FP32 to FP16 (Half-Precision) conversion engine to reduce memory footprint by \~50% while maintaining visual fidelity.
  - **Smart Stability Logic:** Implements proactive VRAM detection and dynamic OOM (Out-of-Memory) protection. The application automatically downsizes tiles and retries processing to prevent crashes.
  - **Threaded Operation:** Designed with an asynchronous background engine to keep the GUI responsive during heavy AI processing, featuring real-time progress tracking and safe cancellation.

## Interface & Comparison

![Application GUI](GUI.png)
*Figure 1: UpscaleLite interface showing real-time tile processing and GPU VRAM detection.*

![Before and After Comparison](be-af.png)
*Figure 2: Visual comparison between original low-resolution image and x4 upscaled result.*

## Installation
1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare the Model:**

      - Download the ONNX model (63.9 MB) from [OpenModelDB](https://openmodeldb.info/models/4x-NomosWebPhoto-esrgan).
      - Rename the file to `4xNomosWebPhoto_esrgan_fp32_opset17.onnx`.
      - Place it inside the `/models/` directory.

3.  **Execute:**

    ```bash
    python src/main.py
    ```

## License & Credits

  - **Software:** Licensed under the **MIT License**.
  - **AI Model:** [4x-NomosWebPhoto-esrgan](https://openmodeldb.info/models/4x-NomosWebPhoto-esrgan) by **Philip Hofmann (Helaman)** (Licensed under **CC-BY-4.0**).
  - *Note: This project does not distribute the model file. Users must download it from the official source.*

-----

*The first execution will perform an automated FP16 conversion, which may take a few moments.*
