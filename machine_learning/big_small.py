x = 1000000000
y = x
bias = 0.000001
learning_times = 1000000

for _ in range(learning_times):
    y += (bias)
y -= x

print(y)


