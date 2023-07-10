import matplotlib.pyplot as plt

# read data from file
with open('plot_log_1.5_8.txt') as f:
    lines = f.readlines()

epochs = []
train_losses = []
test_aucs = []
number_of_organizations = 5
org_contributions = [[] for _ in range(number_of_organizations)]

# parse data
for i in range(0, len(lines), number_of_organizations + 1):
    epoch_line = lines[i]
    epoch = int(epoch_line.split(': ')[1].split(',')[0])
    train_loss = float(epoch_line.split(', ')[0].split(': ')[1])
    test_auc = float(epoch_line.split(', ')[1].split(': ')[1])
    epochs.append(epoch)
    train_losses.append(train_loss)
    test_aucs.append(test_auc)
    for j in range(i+1, i+number_of_organizations + 1):
        org_line = lines[j]
        org_id = int(org_line.split()[1])
        org_contribution = float(org_line.split()[3])
        org_contributions[org_id].append(org_contribution)

# print(epochs)
# print(train_losses)
# print(test_aucs)
print(org_contributions)

# plot graph
plt.figure(figsize=(10, number_of_organizations + 1))
plt.title('Organization Contributions over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Contribution')
plt.xticks(epochs)
for i in range(4):
    plt.plot(epochs, org_contributions[i], label=f'Org {i}')
plt.legend()
# plt.show()
plt.savefig('../figures/plot_log_1.5_8.png')
