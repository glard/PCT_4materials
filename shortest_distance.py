from pymatgen.core.structure import Structure
from pymatgen.core.structure import IStructure


def dist(file):
    crystal = Structure.from_file('data/CIF-DATA/%s.cif' % file)
    super_cell = Structure.from_file('data/CIF-DATA/%s.cif' % file)
    scaling_matrix = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
    super_cell.make_supercell(scaling_matrix=scaling_matrix)
    cd = crystal.as_dict()
    sd = super_cell.as_dict()
    print(cd)
    print(sd)
    shortestDistance = {}
    print([i.as_dict()['abc'] for i in super_cell.sites])
    print([i.as_dict()['abc'] for i in crystal.sites])
    center_indices, points_indices, offset_vectors, distances = crystal.get_neighbor_list(r=10)

    for i, x1 in enumerate(crystal.sites):
        for j, y1 in enumerate(super_cell.sites):
            # if i == j or i > j:
            #     continue
            # print(x1.specie)
            x = str(x1.specie)
            y = str(y1.specie)
            if not (x, y) in shortestDistance:
                shortestDistance[(x, y)] = crystal.distance_matrix[i, j]
            else:
                if crystal.distance_matrix[i, j] < shortestDistance[(x, y)]:
                    shortestDistance[(x, y)] = crystal.distance_matrix[i, j]
                    
    # use from_sites as Convenience constructor to make a Molecule from a list of sites.
    return shortestDistance


sd = dist('mp-19295')
structure_from_cif = Structure.from_file('data/CIF-DATA/mp-27936.cif')
# get coordinates
cartesian_cords = structure_from_cif.cart_coords
scaling_matrix = [[3, 0, 0],[0, 3, 0],[0, 0, 3]]
structure_from_cif.make_supercell(scaling_matrix=scaling_matrix)

IStructure.get_all_neighbors()
print()