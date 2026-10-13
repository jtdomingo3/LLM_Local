# 📦 Models Directory

This directory is intended to store local **GGUF model binaries** for offline inference and local testing.

> **Note:** Binary model weights (`*.gguf`, `*.bin`, etc.) exceed GitHub's 100MB file limit and are intentionally excluded from version control via `.gitignore`.

---

## 📥 How to Download Models

You can download and set up the models using either of the following methods:

### Method 1: Using Ollama (Recommended)

The microservice backend natively connects to Ollama. You can pull the calibrated models with:

```powershell
# 1. Daily driver - general chat & reasoning (3.8B, ~2.5 GB)
ollama pull phi4-mini

# 2. Fast coding assistant (3B, ~2.0 GB)
ollama pull qwen2.5-coder:3b

# 3. Power coding - complex refactoring (7B, ~4.5 GB)
ollama pull qwen2.5-coder:7b
```

Or run the automated setup script from the repository root:
```powershell
powershell -ExecutionPolicy Bypass -File ..\setup_ollama.ps1
```

---

### Method 2: Direct GGUF Download from Hugging Face

If you prefer to keep standalone `.gguf` files directly inside this `models/` folder:

| Model | Recommended Quantization | Hugging Face Repository | Filename Target in this Folder |
|---|---|---|---|
| **Phi-4 Mini (3.8B)** | `Q4_K_M` (~2.49 GB) | [microsoft/Phi-4-mini-instruct-GGUF](https://huggingface.co/microsoft/Phi-4-mini-instruct-GGUF) | `phi4-mini.gguf` |
| **Qwen 2.5 Coder (3B)** | `Q4_K_M` (~1.93 GB) | [Qwen/Qwen2.5-Coder-3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF) | `qwen2.5-coder-3b.gguf` |
| **Qwen 2.5 Coder (7B)** | `Q4_K_M` (~4.68 GB) | [Qwen/Qwen2.5-Coder-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF) | `qwen2.5-coder-7b.gguf` |

#### Importing a Downloaded GGUF into Ollama:
To register a downloaded `.gguf` file with Ollama:
```powershell
# Create a Modelfile pointing to your local GGUF
echo "FROM ./phi4-mini.gguf" > Modelfile

# Create the model in Ollama
ollama create phi4-mini -f Modelfile
```

---

### Method 3: Hardlink Existing Ollama Cache to `models/` (Zero Extra Disk Space)

If you already pulled the models via Ollama and want them visible in this directory without duplicating disk storage:

```powershell
# In PowerShell (Run as Administrator or Developer Mode)
# Locate blob sha in ~/.ollama/models/blobs and link:
New-Item -ItemType HardLink -Path .\models\phi4-mini.gguf -Target "$HOME\.ollama\models\blobs\<blob-hash>"
```
