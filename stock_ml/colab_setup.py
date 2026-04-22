"""
Colab 1-click setup — mount Drive, clone repo, install deps, verify data.

Usage (paste into a Colab cell):
    !git clone https://github.com/nguyenduccanh011/train_ai_ml.git /content/repo
    %cd /content/repo/stock_ml
    %run colab_setup.py
"""
import os
import subprocess
import sys


REPO_URL = "https://github.com/nguyenduccanh011/train_ai_ml.git"
REPO_DIR = "/content/repo"
DRIVE_BASE = "/content/drive/MyDrive/stock_ml_hub"
DATA_DIR = os.path.join(DRIVE_BASE, "portable_data", "vn_stock_ai_dataset_cleaned")
RESULTS_DIR = os.path.join(DRIVE_BASE, "results")


def mount_drive():
    """Mount Google Drive."""
    if os.path.exists("/content/drive/MyDrive"):
        print("[OK] Google Drive already mounted")
        return True

    try:
        from google.colab import drive
        drive.mount("/content/drive")
        print("[OK] Google Drive mounted")
        return True
    except Exception as e:
        print(f"[ERROR] Cannot mount Drive: {e}")
        return False


def setup_drive_dirs():
    """Create Drive directory structure if not exists."""
    dirs = [
        DRIVE_BASE,
        os.path.join(DRIVE_BASE, "portable_data"),
        RESULTS_DIR,
        os.path.join(DRIVE_BASE, "models"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"[OK] Drive directories ready at {DRIVE_BASE}")


def check_data():
    """Check if data exists on Drive."""
    if os.path.isdir(DATA_DIR):
        clean_symbols = os.path.join(DATA_DIR, "clean_symbols.txt")
        all_sym_dir = os.path.join(DATA_DIR, "all_symbols")

        n_symbols = 0
        if os.path.isdir(all_sym_dir):
            n_symbols = len([d for d in os.listdir(all_sym_dir)
                            if d.startswith("symbol=")])

        if n_symbols > 0:
            print(f"[OK] Data found: {n_symbols} symbols in {DATA_DIR}")
            return True
        else:
            print(f"[WARN] Data dir exists but no symbols found: {DATA_DIR}")
            return False
    else:
        print(f"[MISSING] Data not found at {DATA_DIR}")
        print(f"")
        print(f"  Upload instructions:")
        print(f"  1. On local machine, compress your data:")
        print(f"     cd portable_data")
        print(f"     tar -czf vn_stock_ai_dataset_cleaned.tar.gz vn_stock_ai_dataset_cleaned/")
        print(f"  2. Upload vn_stock_ai_dataset_cleaned.tar.gz to Google Drive:")
        print(f"     MyDrive/stock_ml_hub/portable_data/")
        print(f"  3. Then run this cell to extract:")
        print(f"     !cd {os.path.join(DRIVE_BASE, 'portable_data')} && "
              f"tar -xzf vn_stock_ai_dataset_cleaned.tar.gz")
        print(f"  4. Re-run this setup script")
        return False


def clone_or_pull_repo():
    """Clone or pull latest code from GitHub."""
    if os.path.isdir(os.path.join(REPO_DIR, ".git")):
        print("[INFO] Repo exists, pulling latest...")
        subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"],
                       capture_output=True, text=True)
        print("[OK] Repo updated")
    else:
        print(f"[INFO] Cloning {REPO_URL}...")
        subprocess.run(["git", "clone", REPO_URL, REPO_DIR],
                       capture_output=True, text=True)
        print("[OK] Repo cloned")


def install_deps():
    """Install Python dependencies."""
    req_path = os.path.join(REPO_DIR, "stock_ml", "requirements.txt")
    if os.path.exists(req_path):
        print("[INFO] Installing dependencies...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", req_path],
            capture_output=True, text=True,
        )
        print("[OK] Dependencies installed")
    else:
        print("[WARN] requirements.txt not found")


def print_env_info():
    """Print environment summary."""
    print("\n" + "=" * 60)
    print("ENVIRONMENT SUMMARY")
    print("=" * 60)

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            print("  GPU: None (CPU only)")
    except ImportError:
        print("  GPU: torch not installed yet")

    import shutil
    total, used, free = shutil.disk_usage("/content")
    print(f"  Disk free: {free / 1e9:.1f} GB")

    print(f"  Data dir:    {DATA_DIR}")
    print(f"  Results dir: {RESULTS_DIR}")
    print(f"  Repo dir:    {REPO_DIR}")
    print(f"  Working dir: {os.path.join(REPO_DIR, 'stock_ml')}")

    print("\n" + "=" * 60)
    print("READY! Run pipeline with:")
    print(f"  %cd {os.path.join(REPO_DIR, 'stock_ml')}")
    print(f"  !python run_pipeline.py --all          # All active models")
    print(f"  !python run_pipeline.py --version v27  # Single model")
    print("=" * 60)


def main():
    print("=" * 60)
    print("STOCK ML — COLAB SETUP")
    print("=" * 60)

    # Step 1: Mount Drive
    if not mount_drive():
        return

    # Step 2: Setup Drive directories
    setup_drive_dirs()

    # Step 3: Clone/pull repo
    clone_or_pull_repo()

    # Step 4: Install dependencies
    install_deps()

    # Step 5: Check data
    data_ok = check_data()

    # Step 6: Change to working directory
    stock_ml_dir = os.path.join(REPO_DIR, "stock_ml")
    if os.path.isdir(stock_ml_dir):
        os.chdir(stock_ml_dir)
        # Add to Python path
        if stock_ml_dir not in sys.path:
            sys.path.insert(0, stock_ml_dir)

    # Step 7: Print summary
    print_env_info()

    if not data_ok:
        print("\n[ACTION REQUIRED] Upload data to Drive before running pipeline")


if __name__ == "__main__":
    main()
