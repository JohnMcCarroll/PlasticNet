import matplotlib.pyplot as plt

# Block graph script

with open(r'D:\Machine Learning\PlasticNet\research\paths\outputs\gradients.txt', 'r') as file:
    blocks = dict()
    for line in file:
        if line == '\n':
            # index += 1
            continue
        
        fields = line.split(' ')
        key = fields[0] + ' ' + fields[1]
        grad = float(fields[2].strip('grad:'))

        if key in blocks:
            blocks[key].append(grad)
        else:
            blocks[key] = [grad]

index = 0
for key in blocks:
    plt.plot(blocks[key])
    index += 1
    if index > 15:
        break
plt.show()