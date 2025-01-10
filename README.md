# Accident scenario clustering

1. Goal of this research: Development of a model for deriving representative accident scenarios
   - Optimal clustering method and numerous dataset must be prepared
   - Considering clustering method: UMAP, DBSCAN
   - Given dataset: Tcl (i.e., peak cladding temperature)
   - Features of Tcl: Time serial data
   - (n_sample, n_feature) ~= (9000, 5000)
   - One of simulated variable from MAAP5.0 code
   - Very non-linear, highly uncertain concepts are involved
   - Not allowed to analyze the source code

2. Limitations of past clustering researches:
   - Many attempts to use UMAP for data clustering
   - UMAP has been numerously verified to categorize & cluster the image data
   - But, we observed UMAP has poor clustering performance of time serial data
   - The main reason is commonly used metric method of the UMAP
   - Calculation method (i.e., metric) of distance between dataset is commonly euclidean
   - Our mathematic insight reminds us euclidean metric is not proper to compute the differences of time serial data
   - Therefore, we applied soft-DTW (dynamic time warping) method instead of other metric method in UMAP

3. Contribution
   - Normally, soft-DTW requires much more computational load than eucliden
   - Because soft-DTW needs to search & compare the overall shapes of the data
        * Load of euclidean: O(n)
        * Load of soft-DTW: O(nm)
   - To prevent the computationally overwhelmed problem, this study applied "Sliding window method" when soft-DTW searches similarity of the data
   - Therefore, much enhanced clustering performance & shortened computing time is observed
