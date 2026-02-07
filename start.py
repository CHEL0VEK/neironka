import argparse
import subprocess
import sys
from pathlib import Path

from train import main as train_main


def run_streamlit():
    app_path = Path(__file__).with_name("app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Запуск чат-бота.")
    parser.add_argument("--train", action="store_true", help="Обучить модель перед запуском")
    parser.add_argument("--mode", choices=["cli", "web"], default="cli", help="Режим запуска")
    args = parser.parse_args()

    if args.train:
        train_main()

    if args.mode == "web":
        run_streamlit()
    else:
        from chat import main as chat_main

        chat_main()


if __name__ == "__main__":
    main()
