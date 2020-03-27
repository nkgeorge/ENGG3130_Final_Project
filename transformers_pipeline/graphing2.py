import matplotlib.pyplot as plt

win = [0, 1, 0.5, 0.5, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0.5]
date = ['11/02', '11/10', '11/24', '12/01', '12/04', '12/07', '12/15', '12/22', '12/26',
        '12/28', '01/01', '01/11', '01/19', '01/22', '02/01']

plt.figure(figsize=(10,4))
plt.bar(date, win)
plt.ylabel('Win/Draw/Loss')
plt.xlabel('Date')
plt.savefig('winvdate.jpg')
plt.show()