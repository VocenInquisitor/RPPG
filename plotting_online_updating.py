# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:59:17 2025

@author: Sourav
"""
import numpy as np
from matplotlib import pyplot as plt
import pickle

#file1 = "Store_robust_vf_cost_new_format"
#file2 = "Store_robust_cf_cost_new_format" # this pair for MR

#file1 = "Store_robust_vf_garnet_latest"
#file2 = "Store_robust_cf_garnet_latest"


file1 = "Store_robust_vf_garnet_15_20_new_model"
file2 = "Store_robust_cf_garnet_15_20_new_model"

#file1 = "Store_robust_vf_cost_without_oscillation_MR"
#file2 = "Store_robust_cf_cost_without_oscillation_MR"
#env_nm = "MR_4s_2a"
env_nm = "garnet_15s_20a"

file3 = "Epi_RC_objective_"+env_nm
file4 = "Epi_RC_constrainte_"+env_nm

with open(file1,"rb") as f:
    vf_list = pickle.load(f)
f.close()

with open(file2,"rb") as f:
    cf_list = pickle.load(f)
f.close()

with open(file3,"rb") as f:
    epi_vf_list = pickle.load(f)
f.close()

with open(file4,"rb") as f:
    epi_cf_list = pickle.load(f)
f.close()


vf_list = np.array(vf_list)
cf_list = np.array(cf_list)

evf_list = epi_vf_list
ecf_list = epi_cf_list

lambda_ = 10
b = 90

#max_instances = np.where(vf_list >= 119)
#vf_max_points = vf_list[max_instances]
#cf_max_points = cf_list[max_instances]

x = np.arange(1,len(vf_list)+1)
y = np.min([vf_list/lambda_,(b-cf_list)],axis=0)
plt.figure()
#plt.plot(x,y,color="#A020F0",alpha = 0.35)
plt.plot(vf_list,alpha=0.35)
plt.plot(np.cumsum(vf_list)/np.arange(1,1001),linewidth=4,linestyle='--')
#plt.plot(cf_list,alpha=0.6)
#plt.plot(np.cumsum(cf_list)/np.arange(1,1001),linewidth=4,linestyle=':')
plt.plot(evf_list,alpha=0.95)
#plt.plot(ecf_list,alpha=0.96,linestyle='dashed',linewidth=4,color="#919292")
#plt.plot(np.ones(1000)*b,color='#000000',linestyle="-.",linewidth='3')
#plt.scatter(max_instances,vf_max_points,marker='x',color='#00008B')
#plt.scatter(max_instances,cf_max_points,marker='o',color='#FF0000')
#plt.legend(['vf','avg_vf','cf','avg_cf','epi_vf','epi_cf','baseline','max_vf','cf_corresponding to max vf'])
plt.xlabel('Iterations')
plt.ylabel('Expected objective function')
plt.legend(['vf','avg_vf','epi_vf'])
plt.title('Garnet(15,20)')
plt.savefig('RPPG_and_EPI_RC_'+env_nm+'_vf.pdf')
#plt.savefig('RPPG_and_EPI_RC_'+env_nm+'_cf.pdf')
plt.show()

plt.figure()
plt.plot(cf_list,alpha=0.6)
plt.plot(np.cumsum(cf_list)/np.arange(1,1001),linewidth=4,linestyle=':')
plt.plot(ecf_list,alpha=0.96,linestyle='dashed',linewidth=4,color="#919292")
plt.plot(np.ones(1000)*b,color='#000000',linestyle="-.",linewidth='3')
plt.xlabel('Iterations')
plt.ylabel('Expected constraint function')
plt.legend(['cf','avg_cf','epi_cf','baseline'])
plt.title('Garnet(15,20)')
plt.savefig('RPPG_and_EPI_RC_'+env_nm+'_cf.pdf')
plt.show()

