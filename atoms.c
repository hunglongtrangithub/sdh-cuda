#include "atoms.h"
#include <stdlib.h>

void atoms_data_init(atoms_data *atoms, uint64_t box_size) {
  // Seed random number generator
  srand(0);

  // Randomly distribute atoms in the box
  for (uint64_t i = 0; i < atoms->len; i++) {
    atoms->x_pos[i] = (double)rand() / RAND_MAX * box_size;
    atoms->y_pos[i] = (double)rand() / RAND_MAX * box_size;
    atoms->z_pos[i] = (double)rand() / RAND_MAX * box_size;
  }
}

void atoms_data_cleanup(atoms_data *atoms) {
  free(atoms->x_pos);
  free(atoms->y_pos);
  free(atoms->z_pos);
  atoms->x_pos = NULL;
  atoms->y_pos = NULL;
  atoms->z_pos = NULL;
  atoms->len = 0;
}
