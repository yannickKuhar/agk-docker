import os
import multiprocessing as mp

from itertools import product


def run_job(job):
    model, dataset = job
    cmd = f"python3 nsga2.py -m {model} -d {dataset} -g"

    print(f"[INFO] Running: {cmd}")
    exit_code = os.system(cmd)

    print(f"[INFO] Finished {dataset} on {model} with exit code {exit_code}")
    return exit_code


def main():
    NUM_CORES = 60
    # datasets = ["AIDS", "Mutagenicity", "NCI1", "NCI109",
    #             "PROTEINS", "BZR", "COX2", "DHFR","MUTAG", "PTC_FM", "PTC_FR", "PTC_MM",
    #             "OHSU", "REDDIT-BINARY", "IMDB-BINARY", "github_stargazers"]

    datasets = ["clintox", "BBBP"]

    models = ["svc", "ada", "rf"]

    jobs = list(product(models, datasets))

    with mp.Pool(NUM_CORES) as pool:
        results = pool.map(run_job, jobs)

    print("All jobs done.")


if __name__ == "__main__":
    main()
