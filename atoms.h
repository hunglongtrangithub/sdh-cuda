#ifndef ATOM
#define ATOM

#include <stdint.h>

typedef struct atomdesc {
  double x_pos;
  double y_pos;
  double z_pos;
} atom;

typedef struct atom_list {
  double *x_pos;
  double *y_pos;
  double *z_pos;
  uint64_t len;
} atoms_data;

void atoms_data_init(atoms_data *atom, uint64_t box_size);
void atoms_data_cleanup(atoms_data *atoms);

#endif // !ATOM
