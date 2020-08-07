from matplotlib import pyplot as plt

file_pt = 'train_loss_l7.log'
file_tf = 'train_loss_l8.log'
file_l9 = 'train_loss_l9.log'

with open(file_pt, 'r') as f:
    lines_pt = f.readlines()
with open(file_tf, 'r') as f:
    lines_tf = f.readlines()
with open(file_l9, 'r') as f:
    lines_l9 = f.readlines()

loss_pt = list()
loss_tf = list()
loss_l9 = list()
for lpt, ltf, l9 in zip(lines_pt, lines_tf, lines_l9):
    dpt = lpt.split(' ')
    loss_pt.append(float(dpt[-1][:-1]))
    dtf = ltf.split(' ')
    loss_tf.append(float(dtf[-1][:-1]))
    dl9 = l9.split(' ')
    loss_l9.append(float(dl9[-1][:-1]))

x = list(range(1, len(loss_pt) + 1))
plt.plot(x, loss_pt, label='layer7')
plt.plot(x, loss_tf, label='layer8')
plt.plot(x, loss_l9, label='layer9')
plt.xlabel('epoch')
plt.ylabel('training loss')
plt.legend()
plt.show()

