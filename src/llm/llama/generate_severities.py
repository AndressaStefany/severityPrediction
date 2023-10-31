from string import Template
from pathlib import Path


def generate_inference(clear: bool = False):
    mapping = [
        ["_nltk", "/project/def-aloise/$USER/data/data_preprocessed_tokens_v2.json"],
        ["_trunc", "/project/def-aloise/$USER/data/data_preprocessed_tokens_v3.json"],
    ]
    model_name = "meta-llama/Llama-2-13b-chat-hf"
    [id, path_data_json] = mapping[0]
    n_chunks = 10
    n_tokens_infered_max = 7364
    path_file = Path(__file__)
    path_template = (
        path_file.parent.parent.parent.parent
        / "data"
        / "templates"
        / "template_inference.txt"
    )
    with open(path_template) as f:
        template = Template(f.read())
    if clear:
        for f in path_file.parent.rglob("inference_*"):
            f.unlink()
    for i in range(n_chunks):
        with open(path_file.parent / f"inference{id}_{i}", "w") as f:
            f.write(
                template.substitute(
                    id=i,
                    n_chunks=n_chunks,
                    path_data_json=path_data_json,
                    id_name=id,
                    model_name=model_name,
                    n_tokens_infered_max=n_tokens_infered_max,
                )
            )
    with open(path_file.parent / f"launch{id}_all_inference", "w") as f:
        f.write("\n".join([f"sbatch ./inference{id}_{i}" for i in range(n_chunks)]))


def generate_embeddings(clear: bool = False):
    mapping = [
        ["_nltk", "/project/def-aloise/$USER/data/data_preprocessed_tokens_v2.json"],
        ["_trunc", "/project/def-aloise/$USER/data/data_preprocessed_tokens_v3.json"],
        ["_v4_eclipse", "/project/def-aloise/$USER/data/eclipse_72k.json"],
        ["_v4_mozilla", "/project/def-aloise/$USER/data/mozilla_201k.json"],
    ]
    model_name = "meta-llama/Llama-2-13b-chat-hf"
    layers_ids = "(-1,)"
    [id, path_data_json] = mapping[3]
    n_tokens_infered_max = 1000 # 1104 with hello word limit but seems to be not short enough...
    n_chunks = 10
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
        for f in path_file.parent.rglob("embeddings_gen_*"):
            f.unlink()
    for i in range(n_chunks):
        with open(path_folder_out / f"embeddings_gen{id}_{i}", "w") as f:
            f.write(
                template.substitute(
                    id=i,
                    n_chunks=n_chunks,
                    path_data_json=path_data_json,
                    id_name=id,
                    model_name=model_name,
                    layers_ids=layers_ids,
                    n_tokens_infered_max=n_tokens_infered_max,
                )
            )
    with open(path_folder_out / f"launch{id}_all_embeddings_gen", "w") as f:
        f.write(
            "\n".join([f"sbatch /project/def-aloise/rmoine/launches/embeddings/embeddings_gen{id}_{i}" for i in range(n_chunks)])
        )
    with open(path_folder_out / f"cancel{id}_all_embeddings_gen", "w") as f:
        f.write(
            "\n".join([f"scancel -n embeddings_gen{id}_{i}" for i in range(n_chunks)])
        )


def generate_finetune(clear: bool = False):
    mapping = [
        ["_nltk", "/project/def-aloise/$USER/data/data_preprocessed_tokens_v2.json"],
        ["_trunc", "/project/def-aloise/$USER/data/data_preprocessed_tokens_v3.json"],
    ]
    [id, path_data_json] = mapping[1]
    qlora_alpha = 8
    qlora_dropout = 0.1
    qlora_r = 8
    path_file = Path(__file__)
    path_template = (
        path_file.parent.parent.parent.parent
        / "data"
        / "templates"
        / "template_finetune.txt"
    )
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    with open(path_template) as f:
        template = Template(f.read())
    if clear:
        for f in path_file.parent.rglob("finetune_*"):
            f.unlink()
    with open(path_file.parent / f"finetune{id}", "w") as f:
        f.write(
            template.substitute(
                qlora_alpha=qlora_alpha,
                qlora_dropout=qlora_dropout,
                qlora_r=qlora_r,
                path_data_json=path_data_json,
                id_name=id,
                model_name=model_name,
            )
        )


if __name__ == "__main__":
    generate_embeddings()
