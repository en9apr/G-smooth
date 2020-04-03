#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:47:00 2020

@author: andrew
"""
import numpy as np
import csv
import matplotlib.pyplot as plt

def cone_function(x, x0):
    """
    Creates the function to define the exclusion zones.
    """
    # x_norm.shape is (20000, 5, 1)
    x_norm = np.sqrt(np.square(np.atleast_2d(x)[:, None, :] - np.atleast_2d(x0)[None, :, :]).sum(axis=-1)) # Return the sum of the array elements over the last axis.
    norm_jitter = 1e-10
    return (x_norm + norm_jitter) / (r_x0 + s_x0)

tx = np.linspace(0.3, 1.4, 20000)[:,None]

# Load EI
ei = []
with open('ei1.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    line_count = 0
    for row in csv_reader:
        ei.append(row)
fval=np.array(ei).ravel()

# Dummy radii and standard deviation
r_x0 = np.array([0.03609007, 0.00021602]).ravel()
s_x0 = np.array([0.04629545, 0.01139101]).ravel()
pen_locations = np.array([1.4, 1.26683834]).reshape(-1,1)



###############################################################################

# Cone function
cones = cone_function(tx, pen_locations)

# One column of ones
ones_limit = np.ones(np.size(cones,axis=0))[:,None] 

# Add one column of ones to cone function matrix
cones_concatenate_one=np.concatenate((cones, ones_limit), axis=-1) 

# Smooth and clip using norm
cones_smoothed_and_clipped = np.linalg.norm(cones_concatenate_one, -5, axis=-1)

# multiply EI by smoothed and clipped penaliser
fval *= cones_smoothed_and_clipped

###############################################################################





# Plotting
fig, (ax2, ax5, ax1, ax4) = plt.subplots(4,1, figsize=(8,16))

fig.subplots_adjust(bottom=0.05, top=0.9, left=0.1, right=0.95, hspace=0.25)

ax2.plot(tx, fval, color="blue")
ax2b = ax2.twinx()    
for k in range(len(pen_locations)):
    ax2b.axvline(x=pen_locations[k], color="gray", ls="dashed", lw=2, alpha=0.5)
    ax2b.fill_betweenx(np.array([-40, 30]),pen_locations[k]-r_x0[k], pen_locations[k]+r_x0[k], color="gray", alpha=0.15)
ax2b.set_ylim(-0.5, 1.5)
ax2b.set_yticklabels([]) 
ax2b.set_yticks([])
ax2.set_xlim(0.2, 1.5)
ax2.set_xlabel('x')
ax2.set_ylabel('EI(x)')


ax5b = ax5.twinx()  
ax5.plot(tx, cones_smoothed_and_clipped, color="green", lw=2)
for k in range(len(pen_locations)):
    ax5b.axvline(x=pen_locations[k], color="gray", ls="dashed", lw=2, alpha=0.5)
    ax5b.fill_betweenx(np.array([-40, 30]), pen_locations[k]-r_x0[k], pen_locations[k]+r_x0[k], color="gray", alpha=0.15)
ax5b.set_ylim(-0.5, 1.5)
ax5b.set_yticklabels([]) 
ax5b.set_yticks([])        
ax5.set_xlim(0.2, 1.5)    
ax5.set_ylim(-0.2, 1.2)
ax5.set_xlabel('x')
ax5.set_ylabel('cones_smoothed_and_clipped(x)')  

ax1b = ax1.twinx()  
ax1.plot(tx, cones_concatenate_one[:,0], color="green", lw=2)
ax1.plot(tx, cones_concatenate_one[:,1], color="green", lw=2)
ax1.plot(tx, cones_concatenate_one[:,2], color="green", lw=2)
for k in range(len(pen_locations)):
    ax1b.axvline(x=pen_locations[k], color="gray", ls="dashed", lw=2, alpha=0.5)
    ax1b.fill_betweenx(np.array([-40, 30]), pen_locations[k]-r_x0[k], pen_locations[k]+r_x0[k], color="gray", alpha=0.15)
ax1b.set_ylim(-0.5, 1.5)
ax1b.set_yticklabels([]) 
ax1b.set_yticks([])        

ax1.set_xlim(0.2, 1.5)    
ax1.set_ylim(-0.2, 1.2)
ax1.set_xlabel('x')
ax1.set_ylabel('cones_concatenate_one(x)') 

ax4b = ax4.twinx()  
ax4.plot(tx, cone_function(tx, pen_locations), color="green", lw=2)
for k in range(len(pen_locations)):
    ax4b.axvline(x=pen_locations[k], color="gray", ls="dashed", lw=2, alpha=0.5)
    ax4b.fill_betweenx(np.array([-40, 30]), pen_locations[k]-r_x0[k], pen_locations[k]+r_x0[k], color="gray", alpha=0.15)
ax4b.set_ylim(-0.5, 1.5)
ax4b.set_yticklabels([]) 
ax4b.set_yticks([])        
ax4.set_xlim(0.2, 1.5)    
ax4.set_ylim(-0.2, 1.2)
ax4.set_xlabel('x')
ax4.set_ylabel('cones(x)') 

plt.pause(0.005)
plt.show()
plt.tight_layout()
fig.savefig('Episode' + '_' + str(1) + '_' 'xopt'+ '_' + str(3) + '.png', transparent=False)






