# Athena AI

Athena AI is a powerful artificial intelligence system designed for code analysis and text processing. It uses advanced machine learning algorithms to understand and process code in various languages.

## Features

- Code Splitter: Athena AI can split code into chunks for easier analysis. It supports various programming languages including C++, Python, Java, and more.
- Text Splitter: Athena AI can also split large text files into smaller chunks for easier processing and analysis.
- Advanced Machine Learning: Athena AI uses advanced machine learning algorithms to understand and process code and text.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- You have installed the latest version of Python.
- You have a Mac machine with brew installed.
- You have installed Tesseract. If not, you can install it using the following command:

```bash
brew install tesseract
```

## Installing Athena AI

To install Athena AI, follow these steps:

```bash
git clone https://github.com/Transia-RnD/athena-ai.git
cd athena-ai
pip install -r requirements.txt
```

## Using Athena AI

Athena AI provides an easy-to-use interface for interacting with the AI model. Here is an example of how to build an index and initialize the client:

```python
from athenah_ai.indexer import AthenaIndexer
from athenah_ai.client import AthenaClient

# Define the path to your project
path: str = '/Users/darkmatter/projects/transia/athena-ai'

# Initialize the indexer
indexer = AthenaIndexer('local', 'id', 'dist', 'athena', 'v1')

# Build the index
indexer.index_dir(path, ['.'], 'athena')

# Initialize the client
client = AthenaClient('id', 'dist', 'athena')

# Send a prompt to the model and print the response
response = client.prompt("For the Using Athena AI part of the readme include ")
print(response)
```

## Contributing to Athena AI

To contribute to Athena AI, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

## Contact

If you want to contact me you can reach me at `<your_email@domain.com>`.

## License

This project uses the following license: `<license_name>`.