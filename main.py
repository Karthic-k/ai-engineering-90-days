import csv

def load_students(filename):
    students =[]

    with open(filename,"r") as file:
        reader=csv.DictReader(file)

        for row in reader:
            students.append({
                "name":row["name"],
                "marks":int(row["marks"])
            })
    return students
def calculate_average(students):
    total=0

    for student in students:
        total+=student["marks"]

    return total/len(students)

def find_Highest(students):
    highest=0
    for student in students:
        if student["marks"]>highest:
            highest=student["marks"]
    return highest

def find_lowest(students):
    lowest=students[0]["marks"]
    for student in students:
        if(student["marks"]<lowest):
            lowest=student["marks"]

    return lowest
def count_passed(students):
    count=0
    for student in students:
        if student["marks"]>=50:
            count+=1
    return count
students = load_students("students.csv")
average=calculate_average(students)
highest=find_Highest(students)
lowest=find_lowest(students)
passed=count_passed(students)
print("Average Marks: ",average)
print("Highest Marks: ",highest)
print("Lowest Marks: ",lowest)
print("Number of Students Passed: ",passed)

for student in students:
    if student["marks"]>=90:
        print(student["name"]+"- A")
    elif student["marks"]>=80:
        print(student["name"]+"- B")
    elif student["marks"]>=70:
        print(student["name"]+"- C")
    elif student["marks"]>=60:
        print(student["name"]+"- D")
    else:
        print(student["name"]+"- F")
