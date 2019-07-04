
import matplotlib.pyplot as plt
import matplotlib

cuda = True
# DATA
avg_train_losses = []
avg_train_accuracies = [0, 0.25511481046031065, 0.20607361744583808, 0.25042091291334095, 0.27742169683580387, 0.2875837371721779, 0.29214474059293044, 0.2950065029931585, 0.29785713013112886, 0.2995986851482326, 0.30163198403648805]
val_losses = []
val_accuracies = [0, 0.1714498299319728, 0.22335094752186588, 0.26633867832847424, 0.28143221574344024, 0.2871568270165209, 0.2913933430515063, 0.29321550048590866, 0.2933825315840622, 0.2936862244897959, 0.2942936103012634]
print(len(avg_train_losses))
print(len(avg_train_accuracies))
print(len(val_accuracies))
print(len(val_losses))

if cuda:
    matplotlib.use('Agg')


fig = plt.figure()
fig.suptitle('Full Image Features Training', fontsize=20)
plt.xlabel('epoch', fontsize=18)
plt.ylabel('loss', fontsize=16)
plt.plot(range(len(avg_train_losses)), avg_train_losses, 'b', label='train')
plt.plot(range(len(val_losses)), val_losses, 'r', label='val')
plt.legend(loc='best')

if cuda:
    plt.savefig('plots/loss.png')
else:
    plt.show()

fig = plt.figure()
fig.suptitle('Full Image Features Training', fontsize=20)
plt.xlabel('epoch', fontsize=18)
plt.ylabel('accuracy', fontsize=16)
plt.plot(range(len(avg_train_accuracies)),
         avg_train_accuracies, 'b', label='train')
plt.plot(range(len(val_accuracies)), val_accuracies, 'r', label='val')
plt.legend(loc='best')

if cuda:
    plt.savefig('plots/accuracy.png')
else:
    plt.show()
