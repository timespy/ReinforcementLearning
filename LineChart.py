import matplotlib.pyplot as plt


class LineChart:
    def __init__(self):
        self.xs = []
        self.ys = []
        self.styles = []
        self.n_lines = 0

    def add_line(self, x, y, style):
        self.xs.append(x)
        self.ys.append(y)
        self.styles.append(style)
        self.n_lines += 1

    def draw(self):
        for i in range(self.n_lines):
            plt.plot(self.xs[i], self.ys[i], self.styles[i])
        plt.show()
