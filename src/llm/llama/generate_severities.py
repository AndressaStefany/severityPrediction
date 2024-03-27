from string import Template
from pathlib import Path
import itertools as it
import random
from typing import *# type: ignore
try:
    import main as m
except ImportError:
    import llama.main as m
import json


def generate_generic(
    folder_name: str, kwargs: List[Dict[str, Any]], id: str = "", chunking: bool = True,
):
    n_chunks = len(kwargs)
    root = Path(__file__).parent.parent.parent.parent
    root_server = f"/project/def-aloise/rmoine/launches/{folder_name}/"
    path_template = root / "data" / "templates" / f"{folder_name}.txt"
    with open(path_template) as f:
        template_str = f.read()
        template = Template(template_str)
    path_imports = root / "data" / "templates" / "imports.txt"
    with open(path_imports) as f:
        imports = f.read()
    path_folder_out = root / "launches" / folder_name
    path_folder_out.mkdir(exist_ok=True, parents=True)
    print("Writting in ", path_folder_out)
    folder_name += id
    for i in range(n_chunks):
        with open((path_folder_out / f"{folder_name}_{i}").resolve(), "w") as f:
            if chunking:
                kwargs[i]['interval_idx'] = i
                kwargs[i]['n_chunks'] = n_chunks
            f.write(
                template.substitute(
                    imports=imports, **kwargs[i]
                )
            )
    with open(path_folder_out / f"start_all_{id}", "w") as f:
        f.write(
            "#/bin/bash\n"
            + "\n".join(
                [f"sbatch {root_server}/{folder_name}_{i};" for i in range(n_chunks)]
            )
        )
    with open(path_folder_out / f"stop_all_{id}", "w") as f:
        f.write(
            "#/bin/bash\n"
            + "\n".join(
                [
                    f"scancel -n {root_server}/{folder_name}_{i};"
                    for i in range(n_chunks)
                ]
            )
        )


def generate_inference():
    kwargs = {
        "dataset_choice": "",
        "missing_file": "",
        "n_tokens_infered_max": 1000,
        "prompt_id": "official",
        "id_name": "",
    }
    n_chunks=50
    for dataset_choice in ["eclipse_72k", "mozilla_200k"]:
        missing_file = f"inference_{dataset_choice}_missing.json"
        for prompt_id in ["official","alpaca"]:
            for missing_file in ["", missing_file]:
                kwargs["dataset_choice"] = dataset_choice
                kwargs["missing_file"] = missing_file
                kwargs["prompt_id"] = prompt_id
                id = f"_{prompt_id}_{dataset_choice}_{'missing' if missing_file != '' else 'normal'}"
                kwargs["id_name"] = id
                generate_generic(
                    "inference",
                    id=f"_{prompt_id}_{dataset_choice}_{'missing' if missing_file != '' else 'normal'}",
                    kwargs=[kwargs for _ in range(n_chunks)],
                )


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

def generate_finetune_evaluation(target_folder: Path):
    with open(Path(target_folder) / "redo.json") as fp:
        parameters = json.load(fp)
    kwargs = {}
    kwargs["dataset_choice"] = parameters["dataset_choice"]
    kwargs["lora_r"] = parameters["lora_r"]
    kwargs["lora_alpha"] = parameters["lora_alpha"]
    kwargs["lora_dropout"] = parameters["lora_dropout"]
    kwargs["model_name"] = parameters["model_name"]
    kwargs["learning_rate"] = parameters["learning_rate"]
    kwargs["limit_tokens"] = parameters["limit_tokens"]
    kwargs["tr_bs"] = parameters["tr_bs"]
    kwargs["num_train_epochs"] = parameters["num_train_epochs"]
    kwargs["tr_weighted_sampling"] = parameters["tr_weighted_sampling"]
    kwargs["prompt_id"] = parameters["prompt_id"]
    kwargs["resume_from_checkpoint"] = parameters["resume_from_checkpoint"]
    generate_generic(
        folder_name="finetune_eval", kwargs=[kwargs], id=target_folder.stem, chunking=False
    )
    
def generate_finetune(clear: bool = False):
    random.seed(0)
    model_name = "/project/def-aloise/rmoine/cache_dir/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93"# "meta-llama/Llama-2-7b-chat-hf"
    dataset_choices = ["eclipse_72k", "mozilla_200k"]
    learning_rate = [1e-4, 1e-5]
    lora_r = [10, 64, 5]
    weighted = [True, False]
    lora_alpha = [4, 10]
    lora_dropout = [0.1, 0, 0.2]
    tr_weighted_sampling = [False, True]
    prompt_ids = ["official", "alpaca"]
    parameters = cartesian_product_dict(
        learning_rate=learning_rate,
        lora_r=lora_r,
        weighted=weighted,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        tr_weighted_sampling=tr_weighted_sampling,
    )
    print(f"{len(parameters)=}")
    N_tests = 30
    random.shuffle(parameters)
    parameters = parameters[:N_tests]
    print(f"Selecting {len(parameters)}")
    for dataset_choice in dataset_choices:
        for prompt_id in prompt_ids:
            kwargs = [{**e, "prompt_id": prompt_id, "dataset_choice": dataset_choice, "model_name": model_name} for e in parameters]
            
            generate_generic(
                folder_name="finetune",
                kwargs=kwargs,
                id=f"_{dataset_choice}_prompt-{prompt_id}",
                chunking=False
            )


if __name__ == "__main__":
    generate_finetune()
