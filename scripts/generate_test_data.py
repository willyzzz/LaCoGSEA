import numpy as np
import pandas as pd
import os

def generate_simulated_counts(output_path, num_samples=50, num_genes=500):
    """
    Generates a simulated gene expression dataset with semi-realistic counts
    (Not log-transformed, ranging from 0 to thousands).
    """
    np.random.seed(42)
    
    # Generate random genes names
    genes = [f"GENE_{i:03d}" for i in range(num_genes)]
    # Use some real gene names to match with KEGG if needed (optional)
    # But for a general test, generic names are fine.
    
    # Generate counts using a Negative Binomial-ish distribution (Gamma-Poisson)
    # to get that overdispersed look of real RNA-seq data.
    data = {}
    for i in range(num_samples):
        # Base expression level + some noise
        means = np.random.exponential(scale=100, size=num_genes)
        counts = np.random.poisson(lam=means)
        # Add some highly expressed genes
        counts[np.random.randint(0, num_genes, size=10)] *= 50
        data[f"Sample_{i:02d}"] = counts
        
    df = pd.DataFrame(data, index=genes)
    df.to_csv(output_path)
    print(f"Simulated un-logged dataset saved to: {output_path}")
    print(f"Max value in dataset: {df.values.max()}")

if __name__ == "__main__":
    target = os.path.join("lacogsea", "data", "simulated_unlogged.csv")
    generate_simulated_counts(target)
