from datetime import datetime, timedelta
import sys
import joblib

### Start Q1.1 Code ###
def mat_vec_product(A: list[list[int]], b: list[int]) -> list[int]:
  '''
  Conduct matrix-vector multiplication with a given matrix and vector

  Parameters
  ----------
  A: list of lists
     Matrix of (n x n) shape in the form of a list
  b: list
     Vector with n elements

  Returns
  -------
  c: list
     Matrix of shape 

  '''

  # Construct a length-n vector of 0s, to be filled with the results
  n = len(A)
  c = [0 for _ in range(n)]

  # Implement serial matrix-vector multiplication via a for loop over the rows of the argument A

  for idx, row in enumerate(A):
   multiplied_row = []

   for val1, val2 in zip(row, b):
     multiplied_row.append(val1 * val2)

   multiplied_product = 0

   for multiplied_elem in multiplied_row:
     multiplied_product += multiplied_elem

   c[idx] = multiplied_product

  return c
### End Q1.1 Code ###

if __name__ == "__main__":
  start_time = datetime.now()
  A_rand = joblib.load("data/A_large.pkl")
  b_rand = joblib.load("data/b_large.pkl")
  result = mat_vec_product(A_rand, b_rand)
  end_time = datetime.now()
  elapsed_time = end_time - start_time
  sys.stderr.write(str(elapsed_time / timedelta(seconds=1)) + "\n")

