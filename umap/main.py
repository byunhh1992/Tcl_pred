import numpy as np
import umap.umap_ as umap
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt



def get_latent_space(n_neighbors: int=30, min_dist: float=0.1, random_state: int=42) -> np.ndarray:

    import random
    Tcl: np.ndarray = np.load("T_clad_241129_standardscaled.npy")
    umap_model: umap.UMAP = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    latent_space: np.ndarray = umap_model.fit_transform(Tcl)

    return latent_space



def main() -> None:
    
    # Optimal parameter for clustering: n_neighbors=30, min_dist=0.1, random_state=42 (this can be varied)
    latent_space = get_latent_space(n_neighbors=30, min_dist=0.1, random_state=42)

    # Clustered data (by DBSCAN algorithm, this if for visualization)
    dbscan: DBSCAN = DBSCAN(eps=0.5, min_samples=2)
    clusters: np.ndarray = dbscan.fit_predict(latent_space)

    # Save the results
    np.save("latent_space.npy", latent_space)
    np.save("clusters.npy", clusters)



if __name__ == "__main__":
    main()