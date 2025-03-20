#include "atom.h"
#include <stddef.h>
#include <stdlib.h>

void atoms_data_init(atoms_data *atoms, unsigned int box_size) {
  srand(1); // Fixed seed for reproducibility
  for (size_t i = 0; i < atoms->len; i++) {
    atoms->x_pos[i] = ((double)(rand()) / RAND_MAX) * box_size;
    atoms->y_pos[i] = ((double)(rand()) / RAND_MAX) * box_size;
    atoms->z_pos[i] = ((double)(rand()) / RAND_MAX) * box_size;
  }
}
