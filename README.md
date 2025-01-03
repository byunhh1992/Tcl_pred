# Tcl_pred

1. Load raw Tcl (i.e., peak cladding temperature profile) data
   - Data is not provided in here, but Tcl.shape = (num_sample=8989, seq_len=5184)
2. Apply UMAP, Tcl is embedded by 2 dimensional latent space
3. We need to formulate a network system which enables:
   - Predicts the Tcl profile when initial state of Tcl (or very short sequence of Tcl) is given
   - A strong NNs can be used; NN may predict a proper 2 dimensional embedded data when short sequence of Tcl is provided
   - Hence, input_dim depends on the length of input, output_dim=2 (always fixed)
