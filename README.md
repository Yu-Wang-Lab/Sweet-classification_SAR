All the original data are collected from peer-reviewed literature or patent. They are stored as .mol file by downloading their structural information from sci-finder. These information can be found in the Teams sharefolder under Zoey's name.

The .mol files are loaded to AlvaDesc software to generate molecular descriptors (total of 5,666 descriptors) including both 2D (4,179) and 3D (1,487) descriptors. The features that were generated by molecular descriptors included a total of 33 categories, encompassing constitutioinal indices, ring descriptors, topological indices, walk and path counts, connectivity indices, information indices, 2D matrix-based descriptors, 2D autocorrelations, burden eigenvalues, p_vsa-like descriptors, ETA indices, edge adjacency indices, geometrical descriptors, 3D matrix-based descriptors, 3D autocorrelations, RDF descriptors, 3D-morse descriptors, whim descriptors, getaway descriptors, randic molecular profiles, functional group counts, atom-centred fragments, atom-type e-state indices, pharmacophore descriptors, 2D atom pairs, 3D atom pairs, charge descriptors, molecular properties, drug-like indices, CATS 3d descriptors, WHALES descriptors, MDE descriptors, and chirality descriptors (Mauri & Bertola, 2022; Roy, 2020). 

To maintain the most representive features for model development and avoid potential over-fitting, a reduction of features was carried out by eliminating missing values (at least one missing values), constant and nearly constant values (standard deviation less than 0.0001), and highly correlated values (pair correlation larger or equal to 0.85), as referenced before (Goel et al., 2021). As a result, only 703 features were remained and normalized by z-scaling for model development.

The software has been installed on the solo computer in the chemical lab. Feature selections are conducted within the software.


