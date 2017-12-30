import pandas as pd

def calculate_variable_density(df):
  density_dict = {}
  for col in df.columns:
    density_dict[col] = df[col].count()/df.shape[0]
  sorted_density_list = sorted(density_dict.items(), key=lambda x: x[1])
  print("\n")
  for l in sorted_density_list:
    print("{:25} => Density of {:4.2f}%".format(l[0], l[1]*100))


