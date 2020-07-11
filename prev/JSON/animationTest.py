import matplotlib.pyplot as plt

plt.ion()
fig = plt.figure()
plt.show()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

def Rend(i,j,k,l):
    print("Has entered")
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    if i!=[]:
        ax1.plot(i[0],i[1])
    if j!=[]:
        ax2.plot(j[0],j[1])
    if k!=[]:
        ax3.plot(k[0],k[1])
    if l!=[]:
        ax4.plot(l[0],l[1])
    plt.pause(1)
    fig.canvas.draw()       


