import json
from pathlib import Path

import pytest
import test_utils

REPO_ROOT = Path(__file__).parents[3]
SANITY_MODELS_CONFIG = REPO_ROOT.joinpath("ts_scripts", "configs", "sanity_models.json")


def load_models() -> dict:
    with open(SANITY_MODELS_CONFIG) as f:
        models = json.load(f)
    return models


@pytest.fixture(name="model", params=load_models(), scope="module")
def models_to_validate(request, model_store, gen_models, ts_scripts_path):
    model = request.param

    if model["name"] in gen_models:
        from ts_scripts.marsgen import generate_model

        generate_model(gen_models[model["name"]], model_store)

    yield model


@pytest.fixture(scope="module")
def torchserve(model_store):
    test_utils.torchserve_cleanup()

    test_utils.start_torchserve(
        model_store=model_store, no_config_snapshots=False, gen_mar=False
    )

    yield

    test_utils.torchserve_cleanup()


def test_config_snapshotting(model, torchserve, ts_scripts_path, grpc_client_stubs):
    from ts_scripts.sanity_utils import run_grpc_test

    run_grpc_test(model)


def test_models_with_rest(model, torchserve, ts_scripts_path):
    from ts_scripts.sanity_utils import run_rest_test

    run_rest_test(model)
