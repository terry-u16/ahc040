import argparse
import shutil
import subprocess


def run_single(seed: int):
    subprocess.run(
        ["cargo", "build", "--release", "--features", "local"]
    ).check_returncode()
    shutil.move("../target/release/ahc040", "./ahc040")

    input_file = f"./pahcer/in/{seed:04}.txt"
    output_file = "./out.txt"
    err_file = "./err.txt"

    with open(input_file, "r") as i:
        with open(output_file, "w") as o:
            with open(err_file, "w") as e:
                subprocess.run(
                    ["./ahc040"], stdin=i, stdout=o, stderr=e
                ).check_returncode()

    with open(err_file, "r") as e:
        for line in e.readlines():
            print(line.strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, required=True)
    args = parser.parse_args()
    run_single(args.seed)
