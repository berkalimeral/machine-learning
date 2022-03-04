import numpy as np

# One Dimensional Numpy Array
one_dimensional_array = np.array([1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5])
print(one_dimensional_array)

# Two Dimensional Numpy Array
two_dimensional_array = np.array([[6, 5], [11, 7], [4, 8], [5, 15]])
print(two_dimensional_array)

# Array Of The Range
sequence_of_integers = np.arange(2, 11)
print(sequence_of_integers)

# Array that contains random numbers with given size
random_integers_between_50_and_100 = np.random.randint(low=20, high=150, size=(5))
print(random_integers_between_50_and_100)

# Array Of Random Floats Between 0 and 1 with given size
random_floats_between_0_and_1 = np.random.random([6])
print(random_floats_between_0_and_1)

# Basically inreased sample of random floats between 0 and 1
random_floats_between_4_and_5 = random_floats_between_0_and_1 + 4.0
print(random_floats_between_4_and_5)

# Basically we use 3 multiplication to increase 50 and 100 array to 150 and 300 range
random_integers_between_150_and_300 = random_integers_between_50_and_100 * 3
print(random_integers_between_150_and_300)

# Task 1
feature = np.arange(6, 20)
print(feature)
label = feature * 3 + 4 #label = (3)(feature) + 4
print(label)

# Task 2
noise = np.random.random([14]) * 7 + 2
print(noise)
label = label + noise # Label and noise have to same size or you got an error
print(label)