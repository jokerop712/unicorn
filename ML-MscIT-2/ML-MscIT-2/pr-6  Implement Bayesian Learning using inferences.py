# Prior probability of disease
P_D = 0.01

# Likelihoods
P_T_given_D = 0.99          # P(T | D)
P_T_given_not_D = 0.05      # P(T | ~D)


# Calculate marginal probability P(T)
P_not_D = 1 - P_D
P_T = (P_T_given_D * P_D) + (P_T_given_not_D * P_not_D)


# Calculate posterior probability P(D | T)
P_D_given_T = (P_T_given_D * P_D) / P_T


# Print result
print(
    f"The probability of having the disease given a positive test result "
    f"is {P_D_given_T:.2f}"
)


# Optional visualization
import matplotlib.pyplot as plt

labels = ['P(D|T)', 'P(~D|T)']
probabilities = [P_D_given_T, 1 - P_D_given_T]

plt.bar(labels, probabilities, color=['blue', 'orange'])
plt.xlabel('Hypothesis')
plt.ylabel('Probability')
plt.title('Posterior Probability Distribution')
plt.show()
