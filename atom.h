#ifndef ATOM
#define ATOM

#include <stddef.h>

typedef struct atomdesc {
  double x_pos;
  double y_pos;
  double z_pos;
} atom;

typedef struct atom_list {
  double *x_pos;
  double *y_pos;
  double *z_pos;
  size_t len;
} atoms_data;

void atoms_data_init(atoms_data *atom, unsigned int box_size);

#endif // !ATOM
