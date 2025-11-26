Configuration
=============

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

Configuration File Name
-----------------------

.. warning::

    Available only for ``version 0.0.2.dev1``.


By default, SMLE will look for a configuration file named smle.yaml in the current directory.
If you would like to use a different name, a different location, or have multiple configuration files for different configurations, you can set the config_file property of SMLE to the path of your file.
You must assign the filename before calling :meth:`run`.

.. code-block:: python

    app = SMLE()
    app.config_file = "my_file.yaml"
    ...
    app.run()