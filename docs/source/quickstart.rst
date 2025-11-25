Quickstart
===========

This section provides a guided overview of setting up SMLE in a new environment, defining a minimal project structure, and executing an initial experiment through a configuration-driven entrypoint. By the end of this section, you will have a functioning Python script that relies on ``smle.yaml`` for configuration rather than hardcoded parameters.

Installation
------------
SMLE is distributed as a standard Python package and can be installed from PyPI.
It is recommended to use a virtual environment (``venv`` or ``conda``) to keep your project dependencies isolated.

.. code-block:: bash

    pip install smle

Basic Usage Steps
-----------------

The typical SMLE workflow is:

1. Generate a project skeleton and configuration file.
2. Implement your training logic using the SMLE entrypoint.
3. Run the script as a normal Python module while SMLE takes care of configuration
   parsing, logging, and integration with external tools.

1. Initialize a Project
^^^^^^^^^^^^^^^^^^^^^^^

Run the CLI tool to generate a template and config file in the current directory.
This will usually create a basic ``main.py`` script and a default ``smle.yaml`` file.

.. code-block:: bash

    smle init

After running this command, you can immediately open the generated files and adapt them to your model, dataset, and training loop.


2. Configuration (``smle.yaml``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SMLE relies on a simple YAML structure to define hyperparameters, paths, logging options, and integrations.
You can generate a blank template using:

.. code-block:: bash

    smle create yaml

You can configure the ``smle.yaml`` file with the hyperparameters and options for your project. The structure of the ``smle.yaml`` file is:

.. code-block:: yaml

    # ---------------------------------------
    # SMLE Configuration (Modify Carefully)
    # ---------------------------------------

    project: project_name

    logger:
        dir: logger

    wandb:
        entity: your_wandb_account
        key: your_wandb_key

    seed: seed
    device: 'cpu'/'cuda'

    training:
        epochs: n_epochs
        lr: lr
        weight_decay: wd
        batch: batch_size

    testing:
        batch: batch_size

.. tip::
    You can add new sections as needed (for example, ``model``, ``optimizer``, ``scheduler``) and access them via the same keys in ``args``. This makes it easy to manage multiple experiments by maintaining different YAML files.


3. Write Your Code
^^^^^^^^^^^^^^^^^^

You can work on the ``main.py`` script to create your project. Use the ``@app.entrypoint`` decorator. Your configuration variables are automatically passed via ``args`` as a nested dictionary that mirrors the structure of ``smle.yaml``.
Inside ``main``, you never hardcode hyperparameters; instead, you read them from ``args`` so that experiments can be reproduced or changed just by editing YAML.

.. code-block:: python

    from smle import SMLE

    app = SMLE()

    @app.entrypoint
    def main(args):
        # 'args' contains your smle.yaml configurations
        print(f"Training with learning rate: {args['training']['lr']}")

        # Your logic here...

    if __name__ == "__main__":
        app.run()

4. Run It
^^^^^^^^^

Once the code and configuration are in place, you can start an experiment by
running your entrypoint script as a normal Python program:

.. code-block:: bash

    python main.py


During execution, SMLE will:

- Parse ``smle.yaml`` and inject its contents into ``args``.
- Initialize logging so that standard output and configuration are captured.
- Optionally connect to remote trackers (for example, WandB) if enabled in the config.