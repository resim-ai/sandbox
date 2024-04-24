

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


int main(int argc, char *argv[]) {

  const size_t B_PER_GB = pow(2, 30);
  if (argc != 2) {
    fprintf(stderr, "Usage: memory_load [mem in GB]");
    return 1;    
  }    
  size_t buffer_size_gb = atoi(argv[1]);
  char *memoryblock = NULL;

  while (!(memoryblock = (char *)calloc(buffer_size_gb * B_PER_GB, sizeof(char)))) {
    buffer_size_gb--;
  }    
  printf("Allocated %lu GB\n", buffer_size_gb);

  srandom(10);
  size_t size = buffer_size_gb * B_PER_GB;
  while (1) {
    for (size_t ii = 0; ii < size; ++ii) {
       memoryblock[ii] = 'A' + ((memoryblock[2 * ii % size] - 'A') % 26);
    }      
  }    
   return 0;
}  
