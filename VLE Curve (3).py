#!/usr/bin/env python
# coding: utf-8

# In[209]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import seaborn as sns
from matplotlib.widgets import SpanSelector
import math
import scipy
from sympy import symbols, Eq, solve


# In[168]:


# coexistence_temp = np.array([0.4,0.41,0.42,0.43,0.44])
# inverse_gas = np.array([35.40806914,28.16798258,21.91249159,16.245217,11.06553708])
# inverse_liq = np.array([1.331229281,1.361101621,1.39302346,1.436343071,1.480891538])
# dens_coexistence_gas = np.reciprocal(inverse_gas)
# dens_coexistence_liq = np.reciprocal(inverse_liq)
# plt.plot(dens_coexistence_gas,coexistence_temp,'y')
# plt.plot(dens_coexistence_liq,coexistence_temp)


# In[169]:


density_f = np.array([0.786,0.801,0.829,0.856])
density_s = np.array([0.855,0.870,0.880,0.903])
T_sf = np.array([0.45,0.55,0.70,1])

critical_point_density = np.array([0.366])
critical_point_temperature = np.array([0.522])

T_gl = np.array([0.450,0.460,0.470,0.480,0.490,0.495,0.500,0.505,0.515])
density_gas = np.array([0.030, 0.036, 0.05, 0.053, 0.064, 0.07, 0.09, 0.09, 0.28])
density_liq = np.array([0.722,0.72,0.70,0.68,0.65,0.65,0.65,0.64,0.43])

T_vl = np.array([0.415,0.425,0.435,0.445,0.450,0.455,0.460,0.465,0.468])
density_gas2 = np.array([0.0163,0.0173,0.0235,0.0274,0.0285,0.0339,0.0406,0.0473,0.0420])
density_liq2 = np.array([0.763,0.757,0.738,0.725,0.712,0.706,0.698,0.695,0.603])

plt.clf()
fig = plt.figure()
y = T_sf
fig, ax = plt.subplots()
ax.plot(density_f,T_sf,'y')
ax.plot(density_s,T_sf,'g')
ax.plot(density_gas,T_gl,'r')
ax.plot(density_liq,T_gl,'c')
ax.plot(density_gas2,T_vl,'m')
ax.plot(density_liq2,T_vl,'d')
ax.scatter(critical_point_density,critical_point_temperature)
plt.show()

x_data_ordered = np.concatenate((density_gas2,density_gas,density_liq,density_s,density_f))
y_data_ordered = np.concatenate((T_vl,T_gl,T_gl,T_sf,T_sf))


# In[217]:


liq_vle_density = np.concatenate((critical_point_density,density_liq,density_liq2))
liq_vle_temp = np.concatenate((critical_point_temperature,T_gl,T_vl))


vle_liq_df = pd.DataFrame({
    'Densities' : liq_vle_density,
    'Temperatures' : liq_vle_temp})
vle_liq_df_sorted = vle_liq_df.sort_values(by=['Densities'])
vle_liq_df_sorted
vle_liq_den = vle_liq_df_sorted['Densities'].to_numpy()
vle_liq_tem = vle_liq_df_sorted['Temperatures'].to_numpy()

generic_x_data2 = np.linspace(min(vle_liq_den),min(density_f),100)

def liquid_curve(x,a,b,c):
    return a*np.arctan(b*x-c)
plt.clf()
popt2,pcov2 = curve_fit(liquid_curve,vle_liq_den,vle_liq_tem,maxfev = 100000000)
plt.plot(generic_x_data2,liquid_curve(generic_x_data2,*popt2),'b')
sns.scatterplot(x = vle_liq_den, y = vle_liq_tem)
plt.show()

# In[178]:

plt.clf()
gas_vle_density = np.concatenate((density_gas2,density_gas,critical_point_density))
gas_vle_temperature = np.concatenate((T_vl,T_gl,critical_point_temperature))
gas_fit = np.polyfit(gas_vle_density,gas_vle_temperature,4)
gas_curve_function = np.poly1d(gas_fit)

gas_density_range = np.linspace(min(gas_vle_density),max(gas_vle_density),100)
#generic_x_data2 = np.linspace(min(),max(),100)
gas_vle_curve, = plt.plot(gas_density_range,gas_curve_function(gas_density_range))
sns.scatterplot(x = gas_vle_density,y = gas_vle_temperature)
plt.show()

# In[179]:


vle_gas_df = pd.DataFrame({
    'Densities' : gas_vle_density,
    'Temperatures' : gas_vle_temperature})
vle_gas_df_sorted = vle_gas_df.sort_values(by=['Densities'])
#vle_gas_df_sorted.drop([15],inplace=True)
vle_gas_den = vle_gas_df_sorted['Densities'].to_numpy()
vle_gas_tem = vle_gas_df_sorted['Temperatures'].to_numpy()


# In[181]:

plt.clf()
def vapor_curve(x,a,b,c,d):
    return a*np.arctan(b*x-c)+d

popt1,pcov1 = curve_fit(vapor_curve,vle_gas_den,vle_gas_tem,maxfev = 100000000)
plt.plot(gas_density_range,vapor_curve(gas_density_range,*popt1),'b')
sns.scatterplot(x=vle_gas_den,y=vle_gas_tem)
plt.show()

# In[182]:


x_data_ordered = np.concatenate((density_gas2,density_gas,density_liq,density_f))
y_data_ordered = np.concatenate((T_vl,T_gl,T_gl,T_sf))


# In[184]:


df = pd.DataFrame({'Densities' : x_data_ordered,
                    'Temperatures' : y_data_ordered})
df1 = df.sort_values(by=['Densities'])

densities = df1['Densities'].to_numpy()
temperatures = df1['Temperatures'].to_numpy()


# In[19]:


df1


# In[185]:

plt.clf()
x_ordered = np.linspace(0,1,num = 100)
x_ordered
#plt.plot(x,np.poly1d(a)(x))
fit_params = np.polyfit(densities,temperatures,8)
curve_function = np.poly1d(fit_params)
lj_vle_curve, = plt.plot(x_ordered,curve_function(x_ordered))

plt.ylim(0,1)
plt.xlim(0,1)
sns.scatterplot(x=densities,y=temperatures)
plt.xlabel('Density')
plt.ylabel('Temperature')
plt.title('2D LJ VLE Curve')
sns.scatterplot(x=critical_point_density,y=critical_point_temperature)
plt.show()


# In[186]:


x_data = lj_vle_curve.get_xdata()
y_data = lj_vle_curve.get_ydata()
x_data, y_data


# In[22]:

plt.clf()
fig3 = plt.plot(x_data,y_data)
plt.ylim(0,1)
plt.show()

# In[187]:


y_data_incremented = []
y_data_copy = y_data

for i in range(0,99,1):
    y_data = y_data + 0.1
    y_data_incremented.append(y_data)

y_data_incremented.insert(0,y_data_copy)
len(y_data_incremented)
    


# In[188]:


x_data_incremented = []
for j in range (0,100,1):
    x_data_incremented.append(x_data)
    
len(x_data_incremented)


# In[189]:

plt.clf()
plt.scatter(x_data_incremented,y_data_incremented,s=0.01)
plt.show()

# In[190]:


def num_of_sobol_points(number_of_points:int):

    """

    Rounds the number of points you input to the nearest number required to be input to the sobol sequence

    i.e.Rounds to the nearest 2^n where n is an int

    """

    exponent_of_two_to_get_required_number_of_points = math.log(number_of_points)/math.log(2)

    rounded_exponent = round(exponent_of_two_to_get_required_number_of_points,0)

    number_of_sobol_points = 2**rounded_exponent

    return int(number_of_sobol_points)

 

dimension = 2

number_of_points = 15000

points_to_generate = num_of_sobol_points(number_of_points)

sobol_sequence = scipy.stats.qmc.Sobol(dimension) 

sobol_values = sobol_sequence.random(points_to_generate)

plt.clf()
sns.scatterplot(x = sobol_values[:,0], y = sobol_values[:,1])
plt.show()


# In[191]:

plt.clf()
vle_densities = np.concatenate([vle_gas_den,vle_liq_den])
vle_temperatures = np.concatenate([vle_gas_tem,vle_liq_tem])
plt.plot(vle_densities,vle_temperatures)

def func3(x,a,b,c,d):
    return a*x**3 + b*x**2 + c*x + d

generic_x_data3 = np.linspace(min(vle_densities),max(vle_densities),1000)

popt3,pcov3 = curve_fit(func3,vle_densities,vle_temperatures,maxfev = 100000000)
plt.plot(generic_x_data3,func3(generic_x_data3,*popt3),'r')
sns.scatterplot(x = vle_densities,y=vle_temperatures)
plt.show()

# In[192]:

plt.clf()
plt.plot(density_f,T_sf)
def solid_curve(x,a,b,c,d):
    return a*np.exp(b*x)+c

generic_x_data4 = np.linspace(max(vle_liq_den),max(density_f),10000)
popt4,pcov4 = curve_fit(solid_curve,density_f,T_sf,maxfev = 100000000)
plt.plot(generic_x_data4,solid_curve(generic_x_data4,*popt4),'r')
sns.scatterplot(x=density_f,y=T_sf)
plt.show()




# In[195]:


x_graph = np.linspace(min(density_f),max(vle_densities),10000)
g1 = popt2[0]*np.arctan(popt2[1]*x_graph - popt2[2])
g2 = popt4[0]*np.exp(popt4[1]*x_graph)+popt4[2]

triple_point_intersection = pd.DataFrame({
    'vle_gas' : g1,
    'sle_liq' : g2,
    'x_axes' : x_graph
})
triple_point_intersection['difference'] = triple_point_intersection['vle_gas'].sub(triple_point_intersection['sle_liq'],axis=0)
abs(triple_point_intersection['difference']).min()


# In[214]:


index = triple_point_intersection.loc[triple_point_intersection['difference'] == -abs(triple_point_intersection['difference']).min()].index
triple_point_intersection.iloc[index]
triple_x = np.array(triple_point_intersection.iloc[index,2])
triple_y = np.array(triple_point_intersection.iloc[index,1])
np.concatenate((triple_x,triple_y))

# In[218]:

plt.clf()
semi_final_fig = plt.figure()
semi_final_fig, plots = plt.subplots()
plots.plot(gas_density_range,vapor_curve(gas_density_range,*popt1),'b')
plots.plot(generic_x_data2,liquid_curve(generic_x_data2,*popt2),'r')
plots.plot(generic_x_data4,solid_curve(generic_x_data4,*popt4),'y')
plots.scatter(vle_densities,vle_temperatures)
plots.scatter(density_f,T_sf)
plots.scatter(triple_x,triple_y,marker = '^',edgecolor='b',s=100)
plt.plot(0.766577,0.399923)
plt.show()
# In[ ]:
vle_liq_df = pd.DataFrame({
    'Densities' : liq_vle_density,
    'Temperatures' : liq_vle_temp})
vle_liq_df_sorted = vle_liq_df.sort_values(by=['Densities'])
vle_liq_df_sorted
vle_liq_den = vle_liq_df_sorted['Densities'].to_numpy()
vle_liq_tem = vle_liq_df_sorted['Temperatures'].to_numpy()

generic_x_data2 = np.linspace(min(vle_liq_den),min(density_f),100)

def liquid_curve(x,a,b,c,d):
    return a*np.arctan(b*x-c)

plt.clf()
popt2,pcov2 = curve_fit(liquid_curve,vle_liq_den,vle_liq_tem,maxfev = 100000000)
plt.plot(generic_x_data2,liquid_curve(generic_x_data2,*popt2),'b')
sns.scatterplot(x=vle_liq_den,y=vle_liq_tem)
plt.show()


# In[159]:


## Updated VLE dataset and SLE dataset for triple point!
plt.clf()
#gas phase VLE
final_fig = plt.figure()
final_fig, phase_plot = plt.subplots()
phase_plot.plot(gas_density_range,vapor_curve(gas_density_range,*popt1),'b')
plt.show()
#liq phase VLE
critical_density = 0.366
triple_point_density = 0.766577
sobol_df = pd.DataFrame(sobol_values)
sobol_df.columns = ['rho','T']
sobol_df['vle_curve_value_at_rho_of_sobol_point'] = np.where(sobol_df["rho"]<critical_density,vapor_curve(sobol_df["rho"],*popt1),liquid_curve(sobol_df["rho"],*popt2)) # If the density of the point is less than the critical point density, use vapour equation, if not use the liquid equation
sobol_df['vle_curve_value_at_rho_of_sobol_point'] = np.where(sobol_df["rho"]>triple_point_density,solid_curve(sobol_df["rho"],*popt4),sobol_df['vle_curve_value_at_rho_of_sobol_point']) # If the density of the point is greater then the triple point density, use the SLE curve , if not just keep the VLE curve value
sobol_df['T_greater_than_VLE_curve'] = sobol_df["T"]>sobol_df["vle_curve_value_at_rho_of_sobol_point"] # If the temperature of the sobol point less greater than the value at the SLE-VLE curve at that density , then  True, else false

print(sobol_df[sobol_df['T_greater_than_VLE_curve']]) # Print the dataframe where only the T_greater_thabn_VLE_curve is True

plt.clf() # clear any plots
sns.scatterplot(data = sobol_df[sobol_df['T_greater_than_VLE_curve']], x='rho',y = 'T',s=1).set(title = "Datapoints to perform MD") # Plot the points where only the T for the sobol point is greater than the value for T on the curve
plt.show()
# %%