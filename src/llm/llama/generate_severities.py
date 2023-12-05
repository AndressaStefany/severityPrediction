from string import Template
from pathlib import Path
import itertools as it
import random


def generate_inference(clear: bool = False):
    dataset_choices = ["mozilla_200k"]
    n_chunks = 100
    path_file = Path(__file__)
    path_template = (
        path_file.parent.parent.parent.parent
        / "data"
        / "templates"
        / "template_inference.txt"
    )
    use_missing = True
    folder_out = path_file.parent / "launches" / "inference"
    folder_out.mkdir(parents=True, exist_ok=True)
    with open(path_template) as f:
        template = Template(f.read())
    if clear:
        for f in path_file.parent.rglob("inference_*"):
            f.unlink()
    root = "/project/def-aloise/rmoine/launches/inference/"
    Llaunches = ["#!/bin/bash"]
    i = 0
    for dataset_choice in dataset_choices:
        for interval_idx in range(n_chunks):
            name = f"inference_{i}"
            i += 1
            missing_file = '""'
            if use_missing:
                missing_file = f"inference_{dataset_choice}_missing.json"
            with open(folder_out / name, "w") as f:
                f.write(
                    template.substitute(
                        dataset_choice=dataset_choice,
                        n_chunks=n_chunks,
                        interval_idx=interval_idx,
                        missing_file=missing_file,
                    )
                )
            Llaunches.append(name)
    with open(folder_out / f"launch_inference", "w") as f:
        f.write("\n".join([f"sbatch " + root + e+";" if i > 0 else e for i,e in enumerate(Llaunches)]))
    with open(folder_out / f"cancel_inference", "w") as f:
        f.write("\n".join([f"scancel -n " + root + e+";" if i > 0 else e for i,e in enumerate(Llaunches)]))


def generate_embeddings(clear: bool = False):
    choice = 0
    dataset_choices = ["eclipse_72k", "mozilla_200k"]
    n_chunks = 10
    model_name = "meta-llama/Llama-2-13b-chat-hf"
    dataset_choice = dataset_choices[choice]
    limit_tokens = (
        1000  # 1104 with hello word limit but seems to be not short enough...
    )
    path_file = Path(__file__)
    path_folder_out = path_file.parent / "launches" / "embeddings"
    path_template = (
        path_file.parent.parent.parent.parent
        / "data"
        / "templates"
        / "template_embeddings_gen.txt"
    )
    with open(path_template) as f:
        template = Template(f.read())
    if clear:
        for f in path_file.parent.rglob(f"emb_*_{dataset_choice}"):
            f.unlink()
    for i in range(n_chunks):
        with open(path_folder_out / f"emb_{i}_{dataset_choice}", "w") as f:
            f.write(
                template.substitute(
                    dataset_choice=dataset_choice,
                    interval_idx=i,
                    n_chunks=n_chunks,
                    model_name=model_name,
                    limit_tokens=limit_tokens,
                    use_cpu=False,
                )
            )
    with open(path_folder_out / f"start_emb_{dataset_choice}", "w") as f:
        f.write(
            "#/bin/bash\n"
            + "\n".join(
                [
                    f"sbatch /project/def-aloise/rmoine/launches/embeddings/emb_{i}_{dataset_choice}"
                    for i in range(n_chunks)
                ]
            )
        )
    with open(path_folder_out / f"stop_emb_{dataset_choice}", "w") as f:
        f.write(
            "#/bin/bash\n"
            + "\n".join(
                [f"scancel -n emb_{i}_{dataset_choice}" for i in range(n_chunks)]
            )
        )


def cartesian_product_dict(**kwargs):
    keys = kwargs.keys()
    values = kwargs.values()

    # Generate the Cartesian product of the values
    product_values = list(it.product(*values))

    # Create dictionaries using the keys and Cartesian product values
    result = [dict(zip(keys, combination)) for combination in product_values]

    return result


def generate_finetune(clear: bool = False):
    random.seed(0)
    dataset_choices = ["eclipse_72k", "mozilla_200k"]
    learning_rate = [1e-4, 1e-5]
    lora_r = [10, 64, 5]
    weighted = [True, False]
    lora_alpha = [4, 10]
    lora_dropout = [0.1, 0, 0.2]
    tr_weighted_sampling = [False, True]
    parameters = cartesian_product_dict(
        learning_rate=learning_rate,
        lora_r=lora_r,
        weighted=weighted,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        tr_weighted_sampling=tr_weighted_sampling,
    )
    print(f"{len(parameters)=}")
    N_tests = 25
    random.shuffle(parameters)
    parameters = parameters[:N_tests]
    print(f"Selecting {len(parameters)}")
    path_file = Path(__file__)
    path_template = (
        path_file.parent.parent.parent.parent
        / "data"
        / "templates"
        / "template_finetune.txt"
    )
    with open(path_template) as f:
        template = Template(f.read())
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    if clear:
        for f in path_file.parent.rglob("finetune_*"):
            f.unlink()
    i = 0
    Llaunches = ["#!/bin/bash"]
    root = "/project/def-aloise/rmoine/launches/finetune/"
    root_local = path_file.parent / "launches" / "finetune"
    root_local.mkdir(exist_ok=True)
    for dataset_choice in dataset_choices:
        for p in parameters:
            with open(root_local / f"finetune{i}", "w") as f:
                f.write(template.substitute(dataset_choice=dataset_choice, **p))
            Llaunches.append(f" {root}finetune{i}")
            i += 1
    with open(root_local / "launch_all_finetune", "w") as fp:
        fp.write("\n".join(["sbatch" + e+";" if i > 0 else e for i,e in enumerate(Llaunches)]))
    with open(root_local / "cancel_all_finetune", "w") as fp:
        fp.write("\n".join(["scancel -n" + e+";" if i > 0 else e  for i,e in enumerate(Llaunches)]))


if __name__ == "__main__":
    generate_inference()
