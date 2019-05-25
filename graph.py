import matplotlib.pyplot as plt

def main():
    with open("savefile.txt", "r") as file:
        listnums = []
        for line in file:
            listnums.append(float(line))

        x = range(len(listnums))
        y = listnums
        plt.plot(x, y)
        plt.title("Traveling Salesman Genetic Algorithm", loc='center')
        plt.xlabel("Generations")
        plt.ylabel("Best Value")
        plt.show()

if __name__ == '__main__':
	main()
