import os

# Controlla se il modello esiste già, se si carica le le informazioni di loss e accuracy
if os.path.exists('history.txt'):
    with open('history.txt', 'r') as f:
        lines = f.readlines()
        old_test_loss, old_test_acc = float(lines[0].split(':')[1].split(',')[0]), float(lines[0].split(':')[2])
        print(f'Loaded test loss: {old_test_loss}, test accuracy: {old_test_acc}')
else :
    old_test_loss, old_test_acc = 100, 0
    print('No history file found, starting from scratch')