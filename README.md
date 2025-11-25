# SMLE: Simplify Machine Learning Environments

![GitHub stars](https://img.shields.io/github/stars/blkdmr/smle?style=social) ![GitHub forks](https://img.shields.io/github/forks/blkdmr/smle?style=social) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/261c40f69583462baa200aee959bcc8f)](https://app.codacy.com/gh/blkdmr/smle/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

![PyPI version](https://img.shields.io/pypi/v/smle) ![License](https://img.shields.io/github/license/blkdmr/smle) [![PyPI Downloads](https://img.shields.io/pypi/dm/smle.svg?label=downloads&logo=pypi&color=blue)](https://pypi.org/project/smle/)

**Stop writing boilerplate. Start training.**

SMLE is a lightweight Python framework that automates the "boring stuff" in Machine Learning projects. It handles configuration parsing, logging setup, and experiment tracking so you can focus on the model.

## Why SMLE?

* **Auto-Configuration:** `yaml` files are automatically parsed and injected into your entrypoint. No more hardcoded hyperparameters.
* **Instant Logging:** All print statements and configs are automatically captured to local logs and remote trackers.
* **Remote Monitoring:** Native integration with [Weights & Biases (WandB)](https://wandb.ai/) to monitor experiments from anywhere.

### ⚠️ Security & WandB Configuration

When using the **wandb** section for remote logging, your API key is currently read directly from the `smle.yaml` file.

**Crucial:** To prevent exposing your credentials, **do not commit** `smle.yaml` to GitHub or remote storage if it contains your real API key.

* **Recommendation:** Add `smle.yaml` and `*.log` files to your `.gitignore` file immediately.
* **Disable:** You can safely remove the `wandb` section from the YAML file if you do not need remote logging features.

## Installation

```bash
pip install smle
````

## Quick Start

### 1\. Initialize a Project

Run the CLI tool to generate a template and config file:

```bash
smle init
```

### 2\. Write Your Code

Use the `@app.entrypoint` decorator. Your configuration variables are automatically passed via `args`.

```python
from smle import SMLE

app = SMLE()

@app.entrypoint
def main(args):
    # 'args' contains your smle.yaml configurations
    print(f"Training with learning rate: {args['training']['lr']}")

    # Your logic here...

if __name__ == "__main__":
    app.run()
```

### 3\. Run It

```bash
python main.py
```

## Configuration (`smle.yaml`)

SMLE relies on a simple YAML structure. You can generate a blank template using:

```bash
smle create yaml
```

### Configuration File Name

By default, SMLE will look for a configuration file named `smle.yaml` in the current directory. If you would like to use a different name, a different location, or have multiple configuration files for different configurations, you can set the `config_file` property of SMLE to the path of your file. You must assign the filename before calling `run()`.

```python
app = SMLE()
app.config_file = "my_file.yaml"
...
app.run()
```

## Contributing

Contributions are welcome! If you have ideas for improvements, feel free to fork the repository and submit a pull request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Roadmap

### 🚀 High Priority

- **Documentation:** Write comprehensive documentation and examples.
- **Security:** Improve user key management (e.g., WandB key) using `.env` file support.
- **Configuration:** Add support for multiple/layered YAML files.

### 🔮 Planned Features

- **ML Templates:** Automated creation of standard project structures.
- **Model Tools:** Utilities for Neural Network creation, training, and testing.
- **Notifications:** Email notification system for completed training runs.
- **Data Tools:** Data exploration and visualization helpers.
- **Analysis:** Result analysis tools (diagrams, confusion matrices, etc.).
- **Integrations:** Support for TensorBoard and similar tracking tools.
- **Testing:** Comprehensive unit and integration tests for the framework.

