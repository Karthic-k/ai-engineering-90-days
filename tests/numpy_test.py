import numpy as np
import matplotlib.pyplot as plt
marks=np.array([85,92,67,74,89])

print(marks)
print(type(marks))

new_marks=marks+5
print("Marks after adding 5:", new_marks)

print("shape:",marks.shape)
print("Dimensions:",marks.ndim)
print("Data_type:",marks.dtype)

students_data=np.array([
    [85,21],
    [92,22],
    [74,20],
    [89,21]
])

print(students_data)
print("shape:",students_data.shape)
print("Dimensions:",students_data.ndim)

print("\n=== NUMPY INDEXING ===")

print("Arun's Marks:",students_data[0,0])
print("Arun's Age:",students_data[0,1])
print("Rahul's Marks:",students_data[2,0])

print("\n=== NUMPY SLICING ===")

print("First 2 students data:",students_data[0:2])
print("All Marks:",students_data[:,0])
print("All Ages:",students_data[:,1])

print("\n=== Mathematical Calcultaion===")
print("marks+5:",marks+5)
print("Marks-5",marks-5)
print("marks*2:",marks*2)
print("marks/2:",marks/2)

print("Average:",marks.mean())
print("Maximum:",marks.max())
print("Minimum:",marks.min())

print("\n===Axis Example===")

print("Mean by colomn:",students_data.mean(axis=0))
print("Mean by row:",students_data.mean(axis=1))

print("\n===Standard Deviation===")
print("Standard Deviation:",marks.std())

print("\n===Correlation===")
experience = np.array([1, 2, 3, 4, 5])
salary = np.array([30, 35, 42, 48, 55])

correlation_matrix=np.corrcoef(experience, salary)
print(correlation_matrix)

correlation=np.corrcoef(experience, salary)[0,1]
print("correlation:", correlation)

print("\n===Scatter Plot===")

plt.scatter(experience, salary)
plt.xlabel("Experience (years)")
plt.ylabel("Salary")
plt.title("Experience vs Salary")
plt.show()