import matplotlib.pyplot as plt

game_sent = [0.385416667, 0.74015748, 0.357027464, 0.354140127, 0.688581315, 0.878192534, 0.41697417, 0.390596745,
             0.584615385, 0.634114583, 0.437819421, 0.802325581, 0.356913183, 0.333873582, 0.488648649]
date = ['11/02', '11/10', '11/24', '12/01', '12/04', '12/07', '12/15', '12/22', '12/26',
        '12/28', '01/01', '01/11', '01/19', '01/22', '02/01']
win = [0, 1, 0.5, 0.5, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0.5]
position = [10, 7, 9, 9, 6, 5, 6, 8, 8, 5, 5, 5, 5, 5, 7]
points = [13,16,17,18,21,24,25,25,28,31,31,34,34,34,35]

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax1.set_xlabel('Date')
ax1.set_ylabel('Sentiment', color=color)
ax1.set(ylim=(0, 1))
ax1.plot(date, game_sent, color=color)
ax1.tick_params(axis='y', labelcolor=color)

color = 'tab:blue'
ax2.set_ylabel('Points', color=color)  # we already handled the x-label with ax1
ax2.plot(date, points, color=color,)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Sentiment of r/reddevils vs League Points')
fig.set_size_inches(10, 7)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('sentvpoints.jpg')
plt.show()
