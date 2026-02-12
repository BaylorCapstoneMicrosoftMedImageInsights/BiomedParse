import importlib.abc
import sys
import os
from azureml.acft.image.components.olympus.app.main import main

if __name__ == "__main__":
    # This calls the Olympus entry point directly.
    # We are providing the config path and name here to avoid passing them as arguments.
    config_dir = os.path.join(os.getcwd(), "configs")
    sys.argv.extend([
        "--config-path",
        config_dir,
        "--config-name",
        "finetune_biomedparse",
    ])
    main()
