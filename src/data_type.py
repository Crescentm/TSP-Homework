import enum
import numpy as np
import matplotlib.pyplot as plt


## 碎碎念
## 这个配货中心和卸货点应该是个图结构感觉
## 随机生成边得保证至少有每个卸货点一条路径到达配货中心
## 然后在图上跑一个算法找到最短路径
## 抽象一点，这个配货中心和卸货点都使用纯数字表达
## 不考虑在线优化

# 无人机的参数
MAX_CAPACITY = 10
MAX_DISTANCE = 20
MAX_SPEED = 60

# 地图参数
# 将配送中心和卸货点分别编号(从0开始)
# 以默认参数为例，[0...4]为配送中心，[0...19]为卸货点
DEPOT_NUM = 5
TARGET_NUM = 20
MIN = float("inf")
MAX = float("inf")


class Map:
    def __init__(self, depot=DEPOT_NUM, target=TARGET_NUM):
        self._depot = depot
        self._saleman = depot + 2
        self._target = target
        # 先在10x10的范围内生成坐标，再随机生成距离
        coordinates = np.random.uniform(0, 10, size=(depot + target, 2))
        self._depot_coordinates = coordinates[:depot]
        self._target_coordinates = coordinates[depot:]

    @property
    def depot(self):
        return self._depot

    @property
    def target(self):
        return self._target

    @property
    def saleman(self):
        return self._saleman

    def get_t2t_distance(self, start, end):
        return np.linalg.norm(
            self._target_coordinates[start] - self._target_coordinates[end]
        )

    def get_d2t_distance(self, start, end):
        return np.linalg.norm(
            self._depot_coordinates[start] - self._target_coordinates[end]
        )

    def __str__(self):
        return f"Map(Distance: \n{self._distance})"

    def plot(self):
        plt.xlabel("X 轴")
        plt.ylabel("Y 轴")
        plt.scatter(self._depot_coordinates[:, 0], self._depot_coordinates[:, 1], c="r")
        for i in range(len(self._depot_coordinates)):
            plt.text(
                self._depot_coordinates[i][0],
                self._depot_coordinates[i][1],
                f"{i}",
                fontsize=8,
                color="black",
            )
        plt.scatter(
            self._target_coordinates[:, 0], self._target_coordinates[:, 1], c="b"
        )
        for i in range(len(self._target_coordinates)):
            plt.text(
                self._target_coordinates[i][0],
                self._target_coordinates[i][1],
                f"{i}",
                fontsize=8,
                color="black",
            )
        plt.show()


class UrgencyLevel(enum.Enum):
    LOW = 180
    MEDIUM = 90
    HIGH = 30
    DEFAULT = 0


class Order:
    def __init__(self, urgency=UrgencyLevel.DEFAULT, destination=-1):
        if not isinstance(urgency, UrgencyLevel):
            raise ValueError(f"Urgency must be an instance of UrgencyLevel Enum")

        self._urgency = urgency
        self._destination = destination

    @property
    def urgency(self):
        return self._urgency

    @property
    def destination(self):
        return self._destination

    def random_order(self):
        return Order(
            urgency=np.random.choice(
                [level for level in UrgencyLevel if level != UrgencyLevel.DEFAULT]
            ),
            destination=np.random.randint(DEPOT_NUM, DEPOT_NUM + TARGET_NUM),
        )

    def __str__(self):
        return f"Order(Urgency: {self._urgency}, Destination: {self._destination})"


# if __name__ == "__main__":
#     map = Map()
#     print(map._depot_coordinates[4])
#     print(map._target_coordinates[2])
#     print(map.get_d2t_distance(4, 2))
