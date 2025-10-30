import matplotlib.pyplot as plt

# Example data
x = [0, 1, 2, 3, 4, 5]
y1 = [0, 1, 4, 9, 16, 25]      # e.g. quadratic
y2 = [0, 1, 2, 3, 4, 5]        # e.g. linear
y3 = [25, 20, 15, 10, 5, 0]    # e.g. decreasing

# Create base figure and axis
fig, ax1 = plt.subplots()

# Plot first dataset
ax1.plot(x, y1, color='r', label='y1 (red)')
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y1', color='r')
ax1.tick_params(axis='y', labelcolor='r')

# Create second y-axis
ax2 = ax1.twinx()
ax2.plot(x, y2, color='b', label='y2 (blue)')
ax2.set_ylabel('Y2', color='b')
ax2.tick_params(axis='y', labelcolor='b')

# Create third y-axis — offset it to the right
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))  # Move it out a bit
ax3.plot(x, y3, color='g', label='y3 (green)')
ax3.set_ylabel('Y3', color='g')
ax3.tick_params(axis='y', labelcolor='g')

# Optional: combine legends
lines = ax1.get_lines() + ax2.get_lines() + ax3.get_lines()
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

plt.title('Three Y-Axes Sharing One X-Axis')
plt.show()
