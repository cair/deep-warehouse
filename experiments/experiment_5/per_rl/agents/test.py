import numpy as np

if __name__ == "__main__":
    loss = 10.0
    loss_decay = 0.99
    n = 1000
    vec = [0]
    other = []
    means = []
    avg = 0
    for i in range(n):
        # New loss
        loss = loss * loss_decay

        # Average of last and the new
        avg = (other + loss) / 2

        # Add to ther
        other.append(avg)

        # record loss in vectorized version
        vec.append(loss)

        # Do mean of the vectorized version and add
        means.append(np.mean(vec))


    print(len(other), other)
    print(len(means), means)
