import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r"D:\MyLab\SCAND\ahg2library_extracted\state.csv")
plt.figure(figsize=(8, 8))
plt.scatter(df["x"], df["y"], s=0.5, c=df["theta"], cmap="hsv")
plt.axis("equal")
plt.xlabel("x"); plt.ylabel("y")
plt.title("Trajectory (color = theta)")
plt.show()