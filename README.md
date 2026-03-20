# UpscaleLite-FP16 (x4)

High-performance image upscaler optimized for low-to-mid end GPUs using ONNX Runtime and DirectML.

## Features
- **Hardware Acceleration:** Supports Nvidia, AMD, and Intel GPUs via DirectML.
- **VRAM Efficiency:** Automated FP16 conversion to minimize memory footprint.
- **Stability:** Dynamic tiling logic to prevent Out-of-Memory (OOM) errors.

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the Model:**
   - Download the ONNX model (63.9 MB) from [OpenModelDB](https://openmodeldb.info/models/4x-NomosWebPhoto-esrgan).
   - Rename the file to `4xNomosWebPhoto_esrgan_fp32_opset17.onnx`.
   - Place it inside the `/models/` directory.

3. **Execute:**
   ```bash
   python src/main.py
   ```

## License & Credits
- **Software:** Licensed under the **MIT License**.
- **AI Model:** [4x-NomosWebPhoto-esrgan](https://openmodeldb.info/models/4x-NomosWebPhoto-esrgan) by **Philip Hofmann (Helaman)** (Licensed under **CC-BY-4.0**).
- *Note: This project does not distribute the model file. Users must download it from the official source.*

## Support the Project

> [!IMPORTANT]
> **Donation info below is for the Vietnamese community only.**
> *(Thông tin ủng hộ dưới đây chỉ dành cho cộng đồng Việt Nam.)*

If you find this tool helpful, consider supporting the developer:

- **MoMo:** 0939798809
- **MBBank:** 0939798809 - ĐINH NGUYỄN DUY KHANG

---
*The first execution will perform an automated FP16 conversion, which may take a few moments.*
