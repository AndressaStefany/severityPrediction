from string import Template
from pathlib import Path


def generate_inference(clear: bool = False):
    n_chunks = 10
    id = "_trunc"
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
                    path_data_json="/project/def-aloise/$USER/data/data_preprocessed_tokens_v3.json",
                    id_name=id
                )
            )
    with open(path_file.parent / f"launch{id}_all_inference", "w") as f:
        f.write("\n".join([f"sbatch ./inference{id}_{i}" for i in range(n_chunks)]))


if __name__ == "__main__":
    generate_inference()
