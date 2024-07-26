# Athenah AI

Athenah AI is a powerful artificial intelligence system designed for code analysis and text processing. It uses advanced machine learning algorithms to understand and process code in various languages.

## Features

- Code Splitter: Athenah AI can split code into chunks for easier analysis. It supports various programming languages including C++, Python, Java, and more.
- Text Splitter: Athenah AI can also split large text files into smaller chunks for easier processing and analysis.
- Advanced Machine Learning: Athenah AI uses advanced machine learning algorithms to understand and process code and text.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- You have installed the latest version of Python.
- You have a Mac machine with brew installed.
- You have installed Tesseract. If not, you can install it using the following command:

```bash
brew install tesseract
```

## Installing Athenah AI

To install Athenah AI, follow these steps:

```bash
git clone https://github.com/Transia-RnD/athenah-ai.git
cd athenah-ai
pip install -r requirements.txt
```

## Using Athenah AI (Google Cloud Storage)

Athenah AI provides an easy-to-use interface for interacting with the AI model. Here is an example of how to build an index and initialize the client:

First you need to tell athenah where the credentials are:

`export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/credentials.json"`

```python
from athenah_ai.indexer import AthenahIndexer
from athenah_ai.client import AthenahClient

# Define the path to your project
path: str = '/Users/darkmatter/projects/transia/athenah-ai'

# Initialize the indexer (use gcs for storage type)
indexer = AthenahIndexer('gcs', 'id', 'dist', 'athenah', 'v1')

# Build the index
indexer.index_dir(path, ['.'], 'athenah')

# Initialize the client (use /tmp for gcs)
client = AthenahClient('id', '/tmp', 'athenah')

# Send a prompt to the model and print the response
response = client.prompt("For the Using Athenah AI part of the readme include ")
print(response)
```

## Using Athenah AI (Local)

Athenah AI provides an easy-to-use interface for interacting with the AI model. Here is an example of how to build an index and initialize the client:

```python
from athenah_ai.indexer import AthenahIndexer
from athenah_ai.client import AthenahClient

# Define the path to your project
path: str = '/Users/darkmatter/projects/transia/athenah-ai'

# Initialize the indexer
indexer = AthenahIndexer('local', 'id', 'dist', 'athenah', 'v1')

# Build the index
indexer.index_dir(path, ['.'], 'athenah')

# Initialize the client
client = AthenahClient('id', 'dist', 'athenah')

# Send a prompt to the model and print the response
response = client.prompt("For the Using Athenah AI part of the readme include ")
print(response)
```

## Contributing to Athenah AI

To contribute to Athenah AI, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

## Contact

If you want to contact me you can reach me at `<your_email@domain.com>`.

## License

This project uses the following license: `<license_name>`.