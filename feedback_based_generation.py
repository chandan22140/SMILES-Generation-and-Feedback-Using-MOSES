
import threading
import time
import json
from openai import OpenAI

# Import necessary Moses modules
from moses.metrics import (
    fraction_valid,
    fraction_passes_filters,
    SNNMetric,
    FragMetric,
    ScafMetric,
    WassersteinMetric
)
from moses.preprocessing import canonicalize_smiles, remove_invalid
from moses.data import get_dataset, get_statistics
from multiprocessing import Pool
import os
from moses.utils import get_mol, mapper

# Initialize the OpenAI client
client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key="YOUR_API_KEY_HERE"  # Replace with your actual API key
)

# Precompute or load Moses statistics outside the loop to avoid redundant computations
def initialize_moses():
    # Load test datasets and compute statistics
    test = get_dataset('test')  # Load default test set
    ptest = get_statistics('test')  # Precompute test statistics

    test_scaffolds = get_dataset('test_scaffolds')  # Load default scaffold test set
    ptest_scaffolds = get_statistics('test_scaffolds')  # Precompute scaffold test statistics

    train = get_dataset('train')  # Load default train set

    return {
        'test': test,
        'ptest': ptest,
        'test_scaffolds': test_scaffolds,
        'ptest_scaffolds': ptest_scaffolds,
        'train': train
    }

# Initialize Moses statistics
moses_stats = initialize_moses()

# Define the function to call the OpenAI API with a timeout
def call_api_with_timeout(prompt, client, model_name, timeout=5):
    result = {"response": None, "error": None}

    def api_call():
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            result["response"] = response.choices[0].message.content.strip().split("\n")[0]
        except Exception as e:
            result["error"] = str(e)

    api_thread = threading.Thread(target=api_call)
    api_thread.start()

    api_thread.join(timeout=timeout)

    if api_thread.is_alive():
        return None, "Timeout: API call took longer than allowed time"
    else:
        return result["response"], result["error"]

# Define the function to evaluate a molecule using Moses
def evaluate_molecule_with_moses(smiles, moses_stats):
    metrics = {}

    # Initialize a pool for multiprocessing
    pool = Pool(1)  # Using single process for individual molecule evaluation

    # Prepare the generated SMILES
    gen = [smiles]

    # Compute validity
    metrics['valid'] = fraction_valid(gen, n_jobs=pool)

    if metrics['valid'] == 0.0:
        # If the molecule is invalid, other metrics are not meaningful
        pool.close()
        pool.join()
        return metrics

    # Remove invalid SMILES and canonicalize
    gen = remove_invalid(gen, canonize=True)

    # Compute filters
    metrics['Filters'] = fraction_passes_filters(get_mol(gen), pool)

    # Compute logP, SA, QED, weight using RDKit-based Wasserstein metrics
    # Note: Since we're dealing with a single molecule, these will be based on its properties
    # Define functions for the properties
    from rdkit.Chem import Descriptors, QED

    def compute_logP(mol):
        return Descriptors.MolLogP(mol)

    def compute_SA(mol):
        from rdkit.Chem import rdMolDescriptors
        return rdMolDescriptors.CalcNumRotatableBonds(mol)

    def compute_QED_score(mol):
        return QED.qed(mol)

    def compute_weight(mol):
        return Descriptors.MolWt(mol)

    mol = get_mol(gen)[0]

    metrics['logP'] = compute_logP(mol)
    metrics['SA'] = compute_SA(mol)
    metrics['QED'] = compute_QED_score(mol)
    metrics['weight'] = compute_weight(mol)

    # Compute SNN, Frag, Scaf metrics
    # Initialize Moses metric classes with precomputed statistics
    snn_metric = SNNMetric(**{
        'gen': gen,
        'pref': moses_stats['ptest']['SNN'],
        'n_jobs': 1,
        'device': 'cpu',
        'batch_size': 512,
        'pool': pool
    })
    metrics['SNN'] = snn_metric(gen=gen, pref=moses_stats['ptest']['SNN'])

    frag_metric = FragMetric(**{
        'gen': gen,
        'pref': moses_stats['ptest']['Frag'],
        'n_jobs': 1,
        'device': 'cpu',
        'batch_size': 512,
        'pool': pool
    })
    metrics['Frag'] = frag_metric(gen=gen, pref=moses_stats['ptest']['Frag'])

    scaf_metric = ScafMetric(**{
        'gen': gen,
        'pref': moses_stats['ptest']['Scaf'],
        'n_jobs': 1,
        'device': 'cpu',
        'batch_size': 512,
        'pool': pool
    })
    metrics['Scaf'] = scaf_metric(gen=gen, pref=moses_stats['ptest']['Scaf'])

    # Close the pool
    pool.close()
    pool.join()

    return metrics

# Define the function to generate feedback based on metrics
def generate_feedback(metrics):
    feedback_messages = []
    
    if metrics.get("valid", 0.0) < 1.0:
        feedback_messages.append("The molecule is invalid. Please ensure correct valencies and bond saturation.")
    else:
        if metrics.get("QED", 0.0) < 0.7:
            feedback_messages.append("The QED is too low. Try generating a molecule with a higher QED while maintaining validity.")
        if metrics.get("SNN", 0.0) < 0.6:
            feedback_messages.append("Increase structural similarity to the reference. The SNN score is too low.")
        if metrics.get("logP", 0.0) > 3.5:
            feedback_messages.append("Lower the logP value to produce a more drug-like molecule.")
        elif metrics.get("logP", 0.0) < 1.0:
            feedback_messages.append("Increase the logP slightly to produce a more realistic molecule.")
    
    if not feedback_messages:
        feedback_messages.append("The molecule looks good, but try to improve the QED slightly if possible.")
    
    feedback_str = " ".join(feedback_messages)
    return feedback_str

# Specify the model name
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct-fast"

# Load the input data
with open("cleaned_data.json", "r") as f:
    data = json.load(f)

results = []
max_iterations = 3

# Iterate over the entries in the data
for i in range(len(data)):  # Changed to iterate over all entries
    entry = data[i]
    idx = entry["index"]
    logP = entry["logP"]
    qed = entry["qed"]
    SAS = entry["SAS"]
    molecular_description = entry["molecular_description"]
    smiles = entry["smiles"]
    selfies = entry["selfies"]
    print(f"============================ Entry {i} ===============================")
    
    current_prompt = f"""
    You are a chemical informatics assistant. Based on the following molecular data, generate the corresponding SMILES string.
    Input:
    logP: {logP},
    qed: {qed},
    SAS: {SAS},
    molecular_description: "{molecular_description}"
    Output:
    SMILES string only, without any additional text or explanations.
    """

    best_smiles = None
    metrics = {}
    for iteration in range(max_iterations):
        smiles_response, smiles_error = call_api_with_timeout(current_prompt, client, model_name, timeout=30)  # Increased timeout if necessary
        if smiles_error:
            print(f"SMILES Error: {smiles_error}")
            break

        if smiles_response is None:
            print("No response received from the model. Stopping iteration.")
            break

        print(f"Real SMILES: {smiles}")
        print(f"Generated SMILES: {smiles_response}")

        # Evaluate the generated SMILES using Moses
        metrics = evaluate_molecule_with_moses(smiles_response, moses_stats)
        
        # Print achieved scores
        print("Achieved Metrics:")
        for metric, score in metrics.items():
            print(f"  {metric}: {score}")

        # Check acceptance criteria
        if metrics.get("valid", 0.0) == 1.0 and metrics.get("QED", 0.0) > 0.7:
            print("Molecule accepted.")
            best_smiles = smiles_response
            break
        else:
            # Generate feedback based on metrics
            feedback = generate_feedback(metrics)
            current_prompt = f"""
            You are a chemical informatics assistant. Consider the following feedback and regenerate a SMILES string accordingly.
            Previous molecule evaluation:
            {feedback}
            Now, based on the original input:
            logP: {logP},
            qed: {qed},
            SAS: {SAS},
            molecular_description: "{molecular_description}"
            Output a new SMILES string only, without additional explanations.
            """
            print(f"Iteration {iteration+1} feedback: {feedback}")

    results.append({
        "index": idx,
        "final_smiles": best_smiles,
        "metrics": metrics
    })

# Save the results to a JSON file
output_file = "generated_results_with_feedback.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_file}")
