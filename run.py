import os
import subprocess
import sys

_UTF8_ENV = {**os.environ, "PYTHONUTF8": "1"}


def run_gen() -> int:
    return subprocess.call([sys.executable, "-m", "generate.generate"], env=_UTF8_ENV)


def run_training() -> int:
    return subprocess.call([sys.executable, "train/train.py"], env=_UTF8_ENV)


def run_view_data() -> int:
    return subprocess.call([sys.executable, "data/view_data.py"])


def run_test() -> int:
    return subprocess.call([sys.executable, "test.py"], env=_UTF8_ENV)


def main() -> None:
    print("Choose mode:")
    print("1. Run generate")
    print("2. Run train")
    print("3. View data")
    print("4. Test (chat model trained)")
    choice = input("Enter 1 | 2 | 3 | 4: ").strip()

    if choice == "1":
        code = run_gen()
    elif choice == "2":
        code = run_training()
    elif choice == "3":
        code = run_view_data()
    elif choice == "4":
        code = run_test()
    else:
        print("Invalid choice. Please enter 1 | 2 | 3 | 4.")
        return

    if code != 0:
        print(f"Process exited with code {code}.")


if __name__ == "__main__":
    main()
