import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
df = pd.read_csv(r"C:\Users\Daniel.000\Documents\New_research_project\Neural_Net_EOS\models\TrainedNaNs_LowLR\training_data_for_current_ANN.txt",delimiter=" ")
arr = df.to_numpy()
temperature_column = 2
sns.histplot(data = arr[:,temperature_column][arr[:,temperature_column]<0.522],bins=200)
plt.show()