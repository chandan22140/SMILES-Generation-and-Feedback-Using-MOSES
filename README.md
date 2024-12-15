# Project: SMILES String Generation and Evaluation Using OpenAI and MOSES

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Setup and Installation](#setup-and-installation)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Key Functions](#key-functions)
8. [Output](#output)
9. [License](#license)

---

## Overview
This project generates and evaluates SMILES (Simplified Molecular Input Line Entry System) strings using OpenAI's API for molecular generation and MOSES metrics for evaluation. The process integrates molecular property prediction and feedback loops to iteratively improve molecule quality based on predefined metrics.

## Features
- **Molecular Data Parsing**: Reads molecular properties from input JSON files.
- **SMILES Generation**: Uses OpenAI's model to generate SMILES strings based on molecular descriptions.
- **Molecular Evaluation**: Computes validity, drug-likeness (QED), similarity, and other metrics using the MOSES framework.
- **Feedback Loop**: Iteratively refines SMILES strings by generating feedback based on evaluation metrics.
- **Result Saving**: Outputs the best SMILES strings and corresponding metrics to a JSON file.

## Requirements
### Libraries and Frameworks
- Python 3.8+
- OpenAI Python client
- MOSES (Molecular Sets)
- RDKit
- Multiprocessing

### Environment
- Access to OpenAI API (requires an API key).

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install required dependencies:
   ```bash
   pip install openai moses rdkit numpy
   ```

3. Set up your OpenAI API key:
   Replace `YOUR_API_KEY_HERE` in the code with your actual OpenAI API key.

4. Ensure MOSES datasets are properly installed and accessible.

## Usage
1. **Prepare Input Data**: Provide a JSON file named `cleaned_data.json` with the following structure:
   ```json
   [
       {
           "index": 1,
           "logP": 2.5,
           "qed": 0.7,
           "SAS": 3.0,
           "molecular_description": "A small organic molecule.",
           "smiles": "CCO",
           "selfies": "[C][C][O]"
       },
       ...
   ]
   ```

2. **Run the Program**:
   Execute the script to generate and evaluate SMILES strings:
   ```bash
   python <script_name>.py
   ```

3. **View Results**:
   Generated results are saved in `generated_results_with_feedback.json`.

## Project Structure
```
<project_directory>/
├── cleaned_data.json              # Input molecular data file
├── <script_name>.py               # Main script
├── generated_results_with_feedback.json # Output results file
└── README.md                      # Project documentation
```

## Key Functions
### SMILES Generation
- **`call_api_with_timeout(prompt, client, model_name, timeout)`**: Generates SMILES strings using OpenAI API with a timeout mechanism.

### Molecular Evaluation
- **`evaluate_molecule_with_moses(smiles, moses_stats)`**: Computes molecular properties like validity, QED, and structural similarity.

### Feedback Generation
- **`generate_feedback(metrics)`**: Provides detailed feedback to refine molecule generation based on evaluation metrics.

### Initialization
- **`initialize_moses()`**: Loads MOSES datasets and precomputes necessary statistics.

## Output
Results are saved in `generated_results_with_feedback.json` with the following structure:
```json
[
    {
        "index": 1,
        "final_smiles": "CCO",
        "metrics": {
            "valid": 1.0,
            "QED": 0.8,
            "logP": 2.4,
            ...
        }
    },
    ...
]
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

