from math import sqrt
# q = (1, 3)
# p = (2, 5)

# sqrt((1 - 2)^2 + (3 - 5)^2)

plot1 = [1, 3]
plot2 = [2, 5]

euclidean_distance = sqrt((plot1[0] - plot2[0])**2 + (plot1[1] - plot2[1])**2)
print(euclidean_distance)