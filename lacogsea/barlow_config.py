import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _canonicalize_dataset_name(name: str) -> str:
    """
    Normalize dataset name to the canonical strings used across the codebase.
    Returns one of: 'scanb', 'Metabric', 'TCGA', 'TCGA_Lung'.
    """
    s = str(name).strip()
    s_lower = s.lower()
    if s_lower == "scanb":
        return "scanb"
    if s_lower == "metabric":
        return "Metabric"
    if s_lower == "tcga":
        return "TCGA"
    if s_lower == "tcga_lung" or s_lower == "tcgalung":
        return "TCGA_Lung"
    return s


def apply_dataset_config(cfg: dict) -> None:
    """
    Apply dataset-specific configuration overrides.
    """
    ds = _canonicalize_dataset_name(cfg.get("testing_dataset_name", ""))
    cfg["testing_dataset_name"] = ds


def apply_model_config(cfg: dict) -> None:
    """
    Apply model-type-specific defaults.
    """
    pass

config = {
    'seed': 42,
    'batch_size': [256],
    'sample_size' : 500,

    'encoder_first_layer_dim': [512],
    'encoder_num_layers': [1],
    'encoder_output_dim': [2],
    'learning_rate': [5e-3],

    'regularization' : ['l1', 'l2', 'elastic'],
    'l1_regularization_lambda' : [1e-4, 1e-3],
    'l2_regularization_lambda': [1e-3, 1e-2],

    'num_epochs': 800,
    'optimizer' : ['Adam'],
    'object_num': [2],

    'scheduler': 'ReduceLROnPlateau',
    'lr_decay_factor' : 0.1,
    'lr_patience' : 100,
    'min_lr' : 1e-6,
    "model_type": "auto_encoder",

    "beta_vae" : [0.01, 0.5],

    'bulk_scaling_method' : 'minmax', 
    'sc_scaling_method': 'minmax',
    'eval_metric' : 'NA',
    'model_save_path': './model_checkpoints',
    'fra_save_path': './fra_pre',
    'training_dataset_name': 'All',
    'testing_dataset_name': 'scanb',
    'device': device,

    'use_all_hyperparameters' : True,
    'pretrained': False,
    'noise_add' : False,
    'cellularity' : False,
    'cellularity_label_for_bulk': '',
    'correct_indice' : False,

    'transformer' : False,
    'test_size' : 0.2,

    'use_initial_params' : False,
    
    'gsea_gene_set': 'GO',
}

apply_dataset_config(config)
apply_model_config(config)
