# ViralVerse SDXL Turbo GPU smoke test (Kaggle)

This test runs a real SDXL Turbo image inference on a Kaggle GPU session. The GitHub Actions Docker smoke test is CPU-only and does not prove GPU inference.

## 1. Import the notebook

Open the notebook file in the repository:

`notebooks/viralverse_sdxl_gpu_smoke_test.ipynb`

In Kaggle, create/import a notebook from this file. If direct GitHub import is unavailable, download the `.ipynb` file from GitHub and use Kaggle's notebook import/upload option.

## 2. Configure the session

In Notebook Settings:
- Accelerator: GPU
- Internet: On (needed to install packages and download SDXL Turbo weights)

Save the settings and restart the session if Kaggle prompts you.

## 3. Run all cells

The notebook:
1. Clones the `v2-testing` branch.
2. Checks CUDA and reports GPU/VRAM.
3. Installs `gpu_worker/requirements.txt`.
4. Runs `gpu_worker/gpu_smoke_test.py`.
5. Displays `gpu_smoke_test.png` if generation succeeds.

The first run downloads model weights and can take several minutes. Keep the notebook session active until all cells finish.

## 4. What success looks like

The output should include:
- `CUDA available: True`
- A GPU name and total VRAM
- `PASS: SDXL Turbo generated a valid 512x512 PNG.`
- A displayed `gpu_smoke_test.png`

If it fails, share the complete traceback from the first failed cell. Do not paste API tokens or secrets.

## Important limitations

- Kaggle GPU availability and usage limits depend on the account and current platform policy.
- A successful single-image test proves only this SDXL image path on that session's GPU. It does not prove LTX video generation, continuous uptime, public worker networking, or Render integration.
- No Render deployment is performed by this notebook.
