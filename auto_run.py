import os
import random
import multiprocessing as mp
from itertools import product


NAME = "test"


def run_job(job):
    model, dataset, kernel = job

    cmd = f"python run.py -m {model} -n {NAME} -d {dataset} -k {kernel} -p"
    print(f"[INFO] Running: {cmd}")

    exit_code = os.system(cmd)
    print(f"[INFO] Finished {cmd} → exit {exit_code}")

    return exit_code


def main():
    NUM_CORES = 12

    models = ["ada"]
    # models = ["svc", "rf", "ada"]
    # datasets = ["AIDS", "Mutagenicity", "NCI1", "NCI109", "PROTEINS", "BZR", "COX2", "DHFR", "MUTAG",
    #             "PTC_FM", "PTC_FR", "PTC_MM", "OHSU", "REDDIT-BINARY", "IMDB-BINARY",
    #             "github_stargazers"]
    datasets = ["MUTAG"]
    kernels = ["vsko"]

    jobs = list(product(models, datasets, kernels))

    print(f"Total jobs: {len(jobs)}")

    with mp.Pool(NUM_CORES) as pool:
        results = pool.map(run_job, jobs)

    print("All jobs completed.")
    print("Exit codes:", results)


if __name__ == "__main__":
    main()
