import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from statistics import mean
import scipy.stats
from decimal import Decimal

#%%

#list containing fractions in respective elongatation states - taken from Spahn paper, 2015
fraction_in_distinct_state = [0.0833333, 0.0315, 0.0833333, 0.25, 0.3958333, 0.0833333, 0.07291667]    #sum is not exactly 1, i think 0.999
print(sum(fraction_in_distinct_state))
# list containing average time (also called half time, in ms) spent by ribosome in respective fraction. Sum of all = 0.3 seconds
average_time_in_distinct_state = [399.90008*num for num in fraction_in_distinct_state] # 2.5aa / s, or 400ms per elongation cycle
print(sum(average_time_in_distinct_state))   #400.0000

# easy formula: decay constant = 1 / half time
# list containing decay constants for respective half times
all_seven_decay_constants = [ 1/ele for ele in average_time_in_distinct_state]
print(all_seven_decay_constants)

print(sum(all_seven_decay_constants))

# half times for all 8 decay constants.. sanity check whether what I did so far is right
half_lives_decay_constants = [1/ blabla for blabla in all_seven_decay_constants]   #0.69... is ln2
print(sum(half_lives_decay_constants))      # 333.336 ... good!


#%% Generate lists of 1000000 step durations for provided decay factor. List can later be used to be randomly pulled from 
# These lists contain only integer values for now. If I ever want to be more precise, I could ensure that there are also decimal values..

step_durations = 10

def generate_list_with_step_durations(b, step_durations: int = 100):
    duration = []
    for x in range (step_durations):
        condition = True
        counter = 1
        while condition:
            r = random.random()
            if r >= 1-b:
                duration.append(counter)
                condition = False
            else: counter += 1
    return duration

#lists containing step durations for respective elongation step
steps1 = generate_list_with_step_durations(all_seven_decay_constants[0], step_durations=step_durations)
steps2 = generate_list_with_step_durations(all_seven_decay_constants[1], step_durations=step_durations)
steps3 = generate_list_with_step_durations(all_seven_decay_constants[2], step_durations=step_durations)
steps4 = generate_list_with_step_durations(all_seven_decay_constants[3], step_durations=step_durations)
steps5 = generate_list_with_step_durations(all_seven_decay_constants[4], step_durations=step_durations)
steps6 = generate_list_with_step_durations(all_seven_decay_constants[5], step_durations=step_durations)
steps7 = generate_list_with_step_durations(all_seven_decay_constants[6], step_durations=step_durations)

nested_list = [steps1, steps2, steps3, steps4, steps5, steps6, steps7]

bins = np.linspace(0, 1000, 110)
plt.hist(steps3, bins, alpha=0.5, label='step 3', color = "blue")
plt.hist(steps4, bins,  alpha=0.5, label='step 4', color = "green")
plt.legend(loc='upper right')
plt.xlabel("Time (ms)")
plt.ylabel("Frequency")
plt.title("Distribution of 500k steps")
plt.show()

means_nested_list = [np.mean(element) for element in nested_list]
print(sum(means_nested_list))

#%% all six exponential decay distributions

print(all_seven_decay_constants)

xdata = np.linspace(1,1000, 500)
ydata_1 = np.exp(-all_seven_decay_constants[0] * xdata)
ydata_2 = np.exp(-all_seven_decay_constants[1] * xdata)
ydata_3 = np.exp(-all_seven_decay_constants[2] * xdata)
ydata_4 = np.exp(-all_seven_decay_constants[3] * xdata)
ydata_5 = np.exp(-all_seven_decay_constants[4] * xdata)
ydata_6 = np.exp(-all_seven_decay_constants[5] * xdata)
ydata_7 = np.exp(-all_seven_decay_constants[6] * xdata)

# f, axx = plt.subplots(1)
# axx.plot(xdata, ydata)

plt.xlim(0, 600)
plt.xlabel("Time (ms)")
plt.ylabel("Fraction remaining")

plt.plot(xdata, ydata_2, label = "step1")
plt.plot(xdata, ydata_3, "-b", label = "step2")
plt.plot(xdata, ydata_4, label = "step3")
plt.plot(xdata, ydata_5, "-g", label = "step4", zorder = 2)
plt.plot(xdata, ydata_6, "magenta", label = "step5")
plt.plot(xdata, ydata_7, "-c", label = "step6")
plt.plot(xdata, ydata_1, color = "red", label = "step7", zorder = 3)

plt.legend(loc='upper right')




#%% simulate entire elongation cycle for 1kk ribosomes

durations_1kk_entire_cycles = []

# n_elongation_cycles = 5000000
n_elongation_cycles = 500

for x in range(n_elongation_cycles):
    a = 0
    for element in nested_list:
        a += random.choice(element)
    durations_1kk_entire_cycles.append(a)

plt.title("Distribution of 5 million individual elongation cycles")
plt.xlabel("Time (ms)")
plt.ylabel("Frequency")
plt.hist(durations_1kk_entire_cycles, bins = 100, range=[0,2000])

print(np.mean(durations_1kk_entire_cycles))


#%% Simulate translation of 2 ribosomes on a circRNA (without overtaking, but bumping may happen). It works!!

import random

# Define the length of the circRNA, the ribosome number and the translation time (in milliseconds)
circRNA_length = 22     # Specifiy number of codons on circRNA   
num_ribosomes = 2
# translation_time = 2400000  # 40 minutes. Always specify as milliseconds
# translation_time = 40* 60 *1000  # 40 minutes. Always specify as milliseconds
translation_time = int(0.5* 60*1000)  # 40 minutes. Always specify as milliseconds

ribosome_length = 10        # this value should remain at 10 (codons)

# Randomly distribute n ribosomes on circRNA
# Create a list to store the positions of the ribosomes
ribosome_positions = []

# Define a function to check if newly added ribosome overlaps with another ribosome position stored in ribosome_positions
def overlap(position):
    for pos in ribosome_positions:
        if abs(pos - position) < ribosome_length or ((pos+circRNA_length)-position)<ribosome_length or ((position+circRNA_length)-pos)<ribosome_length:
            return True
    return False

# Allocate the ribosomes randomly without overlapping
for i in range(num_ribosomes):  #Here, I can specify the number of ribosomes. 
    while True:
        position = random.randint(0, circRNA_length-1)
        if not overlap(position):
            ribosome_positions.append(position)
            break
        
#Return ribosome variables, each variable containing the position of the respective ribosome
ribosome_vars = []
for i in range(num_ribosomes):
    var_name = f'ribosome{i+1}_position' # generate variable names e.g. ribosome1_position
    globals()[var_name] = ribosome_positions[i] # create global variable with variable name and assign value
    ribosome_vars.append(var_name)

print("Ribosome positions:", ribosome1_position, ribosome2_position)

#####next steop: automate the code below!!###
ribosome1_stepsmade = 0
ribosome2_stepsmade = 0

ribosome1_stallcounter = 0
ribosome2_stallcounter = 0

ribosome1_liststepdurations = []
ribosome2_liststepdurations = []

ribosome1_leftover = 0
ribosome2_leftover = 0

distance = []

for x in range(translation_time):   # milliseconds. Variable defined above
    """
    This loop simulates the translation of 2 ribosomes on a circRNA.
    The ribosomes are not allowed to overtake each other, but they can bump into each other.
    The ribosome that is bumped will stall for 1 ms.
    If the ribosome is bumped again, it will stall for 2 ms, and so on.
    The ribosome will continue translation after the stall time has passed.
    If the no bumping occurs, the ribosome will continue translation.
    """
    
    print("ribosome1_position:", ribosome1_position)
    print("ribosome2_position:", ribosome2_position)

    #pick new step duration for each ribosome, if previous one is finished
    if ribosome1_leftover <= 0:  
        ribosome1_newstepduration = random.choice(durations_1kk_entire_cycles)
        ribosome1_leftover += ribosome1_newstepduration
        ribosome1_liststepdurations.append(ribosome1_newstepduration)
    if ribosome2_leftover <= 0:  
        ribosome2_newstepduration = random.choice(durations_1kk_entire_cycles)
        ribosome2_leftover += ribosome2_newstepduration
        ribosome2_liststepdurations.append(ribosome2_newstepduration)
        
    # check - for ribosome 1 - whether ribosome has the possibility to translocate, and if so, subtract from leftover value
    if  ribosome2_position-ribosome1_position == ribosome_length or (ribosome2_position + circRNA_length)-ribosome1_position ==ribosome_length:
        ribosome1_stallcounter += 1 #milliseconds
    else: 
        ribosome1_leftover -= 1
          
    # check - for ribosome 2 - whether ribosome has the possibility to translocate, and if so, subtract from leftover value
    if  ribosome1_position-ribosome2_position == ribosome_length or (ribosome1_position + circRNA_length)-ribosome2_position ==ribosome_length:
        ribosome2_stallcounter += 1 #milliseconds
    else: 
        ribosome2_leftover -= 1

    # if leftover value <= 0, move ribosome 1 codon further (this code could be integrated at the very start of the for loop as well, or code from the start can go here)
    if ribosome1_leftover <= 0:
        ribosome1_position = (ribosome1_position +1 )%circRNA_length
        ribosome1_stepsmade += 1
    if ribosome2_leftover <= 0:
        ribosome2_position = (ribosome2_position +1 )%circRNA_length
        ribosome2_stepsmade += 1


print(f"ribosome1 steps made: {ribosome1_stepsmade}")
print(f"ribosome2 steps made: {ribosome2_stepsmade}")
print("stalling durations(s): ", ribosome1_stallcounter/1000, ribosome2_stallcounter/1000)

print(ribosome1_liststepdurations[:10])

# reconstructed_ribosome_positions = np.zeros((2, translation_time), dtype=int)
# reconstructed_ribosome_positions[0, :] = ribosome1_position


#%% Simulate speed of n ribosomes on circRNA (without overtaking, but bumping may happen)

def simulate_ribosome_translation(circRNA_length, num_ribosomes, translation_time, ribosome_length, durations_1kk_entire_cycles):

    # Define the length of the circRNA, the ribosome number and the translation time (in milliseconds)
    circRNA_length = 20    # Specifiy number of codons on circRNA   
    num_ribosomes = 2
    translation_time = 2400000  # 40 minutes. Always specify as milliseconds
    
    ribosome_length = 10        # this value should remain at 10 (codons)
    
    # Randomly distribute n ribosomes on circRNA
    # Create a list to store the positions of the ribosomes - values are constantly overwritten
    ribosome_positions = []
    
    # Define a function to check if newly added ribosome overlaps with another ribosome position stored in ribosome_positions
    def overlap(position):
        for pos in ribosome_positions:
            if abs(pos - position) < ribosome_length or ((pos+circRNA_length)-position)<ribosome_length or ((position+circRNA_length)-pos)<ribosome_length:
                return True
        return False
    
    # Allocate the ribosomes randomly without overlapping
    for i in range(num_ribosomes):  #Here, I can specify the number of ribosomes. 
        while True:
            position = random.randint(0, circRNA_length-1)
            if not overlap(position):
                ribosome_positions.append(position)
                break
            
    #Return ribosome variables, each variable containing the position of the respective ribosome
    ribosome_vars = []
    for i in range(num_ribosomes):
        var_name = f'ribosome{i+1}_position' # generate variable names e.g. ribosome1_position
        globals()[var_name] = ribosome_positions[i] # create global variable with variable name and assign value
        ribosome_vars.append(var_name)
    
    print(ribosome_positions)
    
    # Initialize variables for each ribosome
    stepsmade = [0]*num_ribosomes
    stallcounter = [0]*num_ribosomes
    liststepdurations = [[] for _ in range(num_ribosomes)]
    leftover = [0]*num_ribosomes
    
    
    for x in range(translation_time):   # milliseconds. Variable defined above
    
        # Pick new step duration for each ribosome, if previous one is finished
        for i in range(num_ribosomes):
            if leftover[i] <= 0:  
                newstepduration = random.choice(durations_1kk_entire_cycles)
                leftover[i] += newstepduration
                liststepdurations[i].append(newstepduration)
            
            
            # Check whether ribosome has the possibility to translocate, and if so, subtract from leftover value
            ribosome_is_stalling = False
            for j in range(num_ribosomes):    # if num_ribosomes == 0, this block of code will not execute
                if i != j:
                    if ribosome_positions[j] - ribosome_positions[i] == ribosome_length or (ribosome_positions[j] + circRNA_length) - ribosome_positions[i] == ribosome_length:
                        ribosome_is_stalling = True
            
            if ribosome_is_stalling == True:
                stallcounter[i] += 1 #milliseconds
            else:
                leftover[i] -= 1
                
    
            # If leftover value <= 0, move ribosome 1 codon further
            if leftover[i] <= 0:
                ribosome_positions[i] = (ribosome_positions[i] +1 )%circRNA_length
                stepsmade[i] += 1
    
    for i in range(num_ribosomes):
        print(f"Ribosome {i} - steps made: {stepsmade[i]}")
        print(f"Ribosome {i} - seconds stalled: {stallcounter[i]/1000}")
        
    
    print(f"\nsteps made by all ribosomes: {sum(stepsmade)}")

#%% Simulate speed of n ribosomes on circRNA (without overtaking, but bumping may happen) for a 100 times

# import random

# # Define the length of the circRNA, the ribosome number, the translation time (in milliseconds) and the number of times this sumulation should run
# circRNA_length = 175       # Specifiy number of codons on circRNA   
# num_ribosomes = 5         # Specify number of ribosomes on circRNA (will be randomly positioned. Overlaps are not allowed)
# translation_time = 2400000  # 40 minutes. Always specify as milliseconds
# simulation_number = 100       # Specify number of times this simulation should run. Collectice codons translated and seconds stalled are recorded for each simulation
# ribosome_length = 10        # this value should always remain at 10 (codons)

# listallsimulations_collectivecodonstranslated = []
# listallsimulations_collectivesecondsstalled = []

# for x in range(simulation_number):          
    
#     # Randomly distribute n ribosomes on circRNA
#     # Create a list to store the positions of the ribosomes (values for each ribosome are constantly overwritten)
#     ribosome_positions = []
    
#     # Define a function to check if newly added ribosome overlaps with another ribosome position stored in ribosome_positions
#     def overlap(position):
#         for pos in ribosome_positions:
#             if abs(pos - position) < ribosome_length or ((pos+circRNA_length)-position)<ribosome_length or ((position+circRNA_length)-pos)<ribosome_length:
#                 return True
#         return False
    
#     # Allocate the ribosomes randomly without overlapping
#     for i in range(num_ribosomes):  #Here, I can specify the number of ribosomes. 
#         while True:
#             position = random.randint(0, circRNA_length-1)
#             if not overlap(position):
#                 ribosome_positions.append(position)
#                 break
            
#     #Return ribosome variables, each variable containing the position of the respective ribosome
#     ribosome_vars = []
#     for i in range(num_ribosomes):
#         var_name = f'ribosome{i+1}_position' # generate variable names e.g. ribosome1_position
#         globals()[var_name] = ribosome_positions[i] # create global variable with variable name and assign value
#         ribosome_vars.append(var_name)
    
#     print(ribosome_positions)
    
#     # Initialize variables for each ribosome
#     stepsmade = [0]*num_ribosomes
#     stallcounter = [0]*num_ribosomes
#     liststepdurations = [[] for _ in range(num_ribosomes)]
#     leftover = [0]*num_ribosomes
    
    
#     for x in range(translation_time):   # milliseconds. Variable defined above
    
#         # Pick new step duration for each ribosome, if previous one is finished
#         for i in range(num_ribosomes):
#             if leftover[i] <= 0:  
#                 newstepduration = random.choice(durations_1kk_entire_cycles)
#                 leftover[i] += newstepduration
#                 liststepdurations[i].append(newstepduration)
            
#             # Check whether ribosome has the possibility to translocate, and if so, subtract from leftover value
#             ribosome_is_stalling = False
#             for j in range(num_ribosomes):    # if num_ribosomes == 0, this block of code will not execute
#                 if i != j:
#                     if ribosome_positions[j] - ribosome_positions[i] == ribosome_length or (ribosome_positions[j] + circRNA_length) - ribosome_positions[i] == ribosome_length:
#                         ribosome_is_stalling = True
            
#             if ribosome_is_stalling == True:
#                 stallcounter[i] += 1 #milliseconds
#             else:
#                 leftover[i] -= 1
    
    
#             # If leftover value <= 0, move ribosome 1 codon further
#             if leftover[i] <= 0:
#                 ribosome_positions[i] = (ribosome_positions[i] +1 )%circRNA_length
#                 stepsmade[i] += 1
                
#     listallsimulations_collectivecodonstranslated.append(sum(stepsmade))
#     listallsimulations_collectivesecondsstalled.append(sum(stallcounter)/1000)
    
# print("for 10 simulations - codons translated (top list) and seconds stalled (bottom list):")
# print(listallsimulations_collectivecodonstranslated[:10])
# print(listallsimulations_collectivesecondsstalled[:10])