from pydrake.multibody.plant import MultibodyPlant

from cenic_bench.parsing import PACKAGE_ROOT, make_parser


def test_make_parser_loads_all_models():
    model_files = sorted((PACKAGE_ROOT / "models").rglob("*.sdf"))
    assert model_files

    for model_file in model_files:
        plant = MultibodyPlant(time_step=0.0)
        relative_path = model_file.relative_to(PACKAGE_ROOT).as_posix()
        models = make_parser(plant).AddModelsFromUrl(
            f"package://cenic_bench/{relative_path}"
        )
        assert models, model_file
