import csv

num_attributes = 6
a = []

print("\nThe Given Training Data Set\n")

with open('enjoysport.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        a.append(row)
        print(row)

print("\nThe initial value of hypothesis:")
hypothesis = ['0'] * num_attributes
print(hypothesis)

# Initialize hypothesis with the first training example
for j in range(num_attributes):
    hypothesis[j] = a[0][j]

print("\nFind S: Finding a Maximally Specific Hypothesis\n")

for i in range(len(a)):
    if a[i][num_attributes] == 'yes':
        for j in range(num_attributes):
            if a[i][j] != hypothesis[j]:
                hypothesis[j] = '?'
            else:
                hypothesis[j] = a[i][j]

        print(
            "For Training instance No:{0} the hypothesis is".format(i),
            hypothesis
        )

print("\nThe Maximally Specific Hypothesis for the given Training Examples:\n")
print(hypothesis)
