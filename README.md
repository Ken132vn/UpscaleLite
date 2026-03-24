# Upscale Lite x4 GUI

High-performance image upscaler optimized for low-to-mid end GPUs using ONNX Runtime and DirectML.

Upscale Lite x4 GUI is a lightweight, portable solution designed to provide high-quality image magnification while maintaining extreme stability on hardware with limited resources, such as integrated graphics (Intel Iris Xe/UHD, AMD Radeon) and entry-level dedicated GPUs.

## Interface & Comparison

![Application GUI](GUI.png)
*Figure 1: Upscale Lite x4 GUI interface showing real-time tile processing and GPU VRAM detection.*

![Before and After Comparison](be-af.png)
*Figure 2: Visual comparison between original low-resolution image and x4 upscaled result.*

## Features

  - **Built-in Native GUI:** A clean, distraction-free interface powered by Tkinter. It offers a seamless "point-and-click" experience for both single-file and batch processing without the overhead of heavy web-based frameworks.
  - **Universal Hardware Acceleration:** Leverages the DirectML execution provider to support cross-vendor hardware acceleration. High-speed inference is available natively on Nvidia, AMD, and Intel silicon without the need for complex driver configurations.
  - **Advanced VRAM Efficiency:** Includes an automated FP32 to FP16 (Half-Precision) conversion engine. This optimization reduces the memory footprint by approximately 50%, enabling high-quality upscaling on devices that typically struggle with AI workloads.
  - **Smart Stability Logic:** Implements proactive VRAM detection and dynamic Out-of-Memory (OOM) protection. The application automatically calculates safe tile sizes and executes a recursive retry mechanism if memory limits are reached, preventing application crashes during heavy processing.
  - **Asynchronous Threaded Engine:** Built on a non-blocking background architecture. The GUI remains fully responsive during inference, providing real-time progress updates, ETA tracking, and safe operation cancellation.

## Technical Requirements

  - **Operating System:** Windows 10 or 11 (64-bit).
  - **GPU:** DirectX 12 compatible (Intel Iris Xe, AMD Radeon, Nvidia GeForce).
  - **Backend:** ONNX Runtime with DirectML execution provider.

## Installation

### For Users (Portable)

1.  Download the latest release package from the **Releases** tab.
2.  Extract the ZIP archive.
3.  Place your `.onnx` model files in the `models/` folder.
4.  Run `Upscale_Lite_x4_GUI.exe`.

### For Developers (Source)

1.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2.  Prepare the Model:

      - Download the ONNX model (63.9 MB) from [OpenModelDB](https://openmodeldb.info/models/4x-NomosWebPhoto-esrgan).
      - Rename the file to `4xNomosWebPhoto_esrgan_fp32_opset17.onnx`.
      - Place it inside the `/models/` directory.

3.  Execute:

    ```bash
    python Upscale_Lite_x4_GUI.py
    ```

## License & Credits

  - **Software:** Licensed under the **MIT License**.
  - **AI Model:** Based on [4x-NomosWebPhoto-esrgan](https://openmodeldb.info/models/4x-NomosWebPhoto-esrgan) by Philip Hofmann (Helaman), licensed under **CC-BY-4.0**.
  - *Note: This project does not distribute the model file. Users must download it from the official source provided in the installation steps.*

-----

*Note: The initial execution will perform an automated FP16 conversion to optimize performance for your specific hardware. This process may take a few moments and will only occur once per model.*
