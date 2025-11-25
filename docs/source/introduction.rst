Introduction
============

Why SMLE?
---------

* **Auto-Configuration:** ``yaml`` files are automatically parsed and injected into your entrypoint, so you avoid hardcoded hyperparameters.
* **Instant Logging:** All print statements and configurations are captured to local logs and compatible remote trackers.
* **Remote Monitoring:** Native integration with `Weights & Biases (WandB) <https://wandb.ai/>`_ lets you monitor experiments from anywhere.

.. tip::
    If you are using SMLE for the first time, start with the ``Quickstart`` section to scaffold a project, configure your first ``smle.yaml``, and run an experiment end-to-end in a few minutes.

Contributing
------------

Contributions are welcome! If you have ideas for improvements, feel free to fork the repository and submit a pull request.

#. Fork the Project
#. Create your Feature Branch (``git checkout -b feature/AmazingFeature``)
#. Commit your Changes (``git commit -m 'Add some AmazingFeature'``)
#. Push to the Branch (``git push origin feature/AmazingFeature``)
#. Open a Pull Request

Roadmap
-------

🚀 High Priority
^^^^^^^^^^^^^^^^

High-priority goals include richer documentation, safer key management (for example, through ``.env`` support), and multiple or layered YAML configurations.

* **Documentation:** Write comprehensive documentation and examples.
* **Security:** Improve user key management (e.g., WandB key) using ``.env`` file support.
* **Configuration:** Add support for multiple/layered YAML files.

🔮 Planned Features
^^^^^^^^^^^^^^^^^^^

Planned features include ML project templates, model utilities, notification tools, data exploration helpers, analysis utilities (such as confusion matrices), additional integrations like TensorBoard, and comprehensive testing support.

* **ML Templates:** Automated creation of standard project structures.
* **Model Tools:** Utilities for Neural Network creation, training, and testing.
* **Notifications:** Email notification system for completed training runs.
* **Data Tools:** Data exploration and visualization helpers.
* **Analysis:** Result analysis tools (diagrams, confusion matrices, etc.).
* **Integrations:** Support for TensorBoard and similar tracking tools.
* **Testing:** Comprehensive unit and integration tests for the framework.
