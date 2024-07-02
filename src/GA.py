import random
from data_type import *
import time


class Pop:
    def __init__(self, route, breakp):
        self._route = route
        self._breakp = breakp

    @property
    def route(self):
        return self._route

    @property
    def breakp(self):
        return self._breakp

    def __str__(self):
        return f"Pop {self._route}, {self._breakp}\n"


class GA:
    def __init__(self, population_size: int, map: Map, orders: list, num_iter=1e3):
        self._salesman_num = map.saleman
        self._map = map
        self._orders = orders
        self._population_size = population_size

        self._population_route = np.empty((population_size, map.target), int)
        self._population_break = np.empty((population_size, map.saleman - 1), int)
        self._population = np.empty(population_size, dtype=object)

        self.total_distance = np.zeros(self._population_size)

        self.iter = 0  # 迭代次数
        self.num_iter = num_iter  # 最高迭代次数
        self.cross_rate = 0.8  # 交叉概率
        self.cross_ox_prob = 0.5  # OX和PMX交叉概率
        self.mutate_rate = 0.3  # 变异概率
        self.mutate_break_rate_pop = 0.2  # 一个个体break变异概率
        self.cost_list = []

    # the break is start index break
    def randbreak(self, route_len, break_len):
        if break_len > self._salesman_num - 1:
            raise ValueError("break_len must be less than salesman_num - 1")
        breaks: list = sorted(
            [random.randint(0, route_len - 1) for _ in range(break_len)]
        )
        while len(breaks) < self._salesman_num - 1:
            breaks.append(-1)
        return breaks

    def init_population(self):
        for i in range(self._population_size):
            route = np.random.permutation(self._map.target)
            self._population_route[i] = route
            break_point = self.randbreak(route.shape[0], self._salesman_num - 1)
            self._population_break[i] = break_point
            self._population[i] = Pop(route, break_point)

    def get_range(self, route, breaks):
        route_ranges = []
        start = 0
        for end in breaks:
            if end < 0:
                # 遇到padding的情况
                break
            route_range = route[start:end]
            if len(route_range) > 0:
                route_ranges.append(route_range)
            start = end
        route_ranges.append(route[start:])
        return route_ranges

    def get_range_cost(self, route, breaks):
        route_ranges = self.get_range(route, breaks)
        cost = 0
        for route_range in route_ranges:
            for i in range(len(route_range) - 1):
                cost += self._map.get_t2t_distance(route_range[i], route_range[i + 1])
        return cost

    def get_best_break_cost(self, route, breaks):
        cost = 0

        ranges = self.get_range(route, breaks)
        for rang in ranges:
            best_distance = 20
            for depot in range(self._map.depot):
                diss = self._map.get_d2t_distance(depot, rang[0])
                dise = self._map.get_d2t_distance(depot, rang[-1])
                dis = diss + dise
                if dis < best_distance:
                    best_distance = dis
            cost += best_distance

        return cost

    def get_one_cost(self, pop):
        route = pop.route
        breaks = pop.breakp
        t2t_cost = self.get_range_cost(route, breaks)
        d2t_cost = self.get_best_break_cost(route, breaks)
        distance = d2t_cost + t2t_cost
        return distance

    def get_cost(self):
        # 计算所有个体的总距离
        for i in range(self._population_size):
            route = self._population_route[i]
            breaks = self._population_break[i]
            t2t_cost = self.get_range_cost(route, breaks)
            d2t_cost = self.get_best_break_cost(route, breaks)
            distance = d2t_cost + t2t_cost
            self.total_distance[i] = distance
        return self.total_distance

    def get_best_depot(self, route, breaks):
        best_depots = []

        ranges = self.get_range(route, breaks)
        for rang in ranges:
            best_distance = 20
            best_depot = 0
            for depot in range(self._map.depot):
                diss = self._map.get_d2t_distance(depot, rang[0])
                dise = self._map.get_d2t_distance(depot, rang[-1])
                dis = diss + dise
                if dis < best_distance:
                    best_distance = dis
                    best_depot = depot
            best_depots.append(best_depot)

        return best_depots

    def get_best_chrom(self, num=8):
        # 找到种群中最优的多个个体
        tmp = self.get_cost()
        index = np.argpartition(tmp, num)[:num]
        return index

    def cross_ox(self, chrom1, chrom2):
        # 只顺序交叉route
        route1 = chrom1.route.tolist()
        route2 = chrom2.route.tolist()

        index1, index2 = random.randint(0, len(route1) - 1), random.randint(
            0, len(route1) - 1
        )
        if index1 > index2:
            index1, index2 = index2, index1

        temp_gene1 = route1[index1:index2]
        temp_gene2 = route2[index1:index2]

        child_route1, child_route2 = [], []
        child_p1, child_p2 = 0, 0

        for i in route2:
            if child_p1 == index1:
                child_route1.extend(temp_gene1)
                child_p1 += 1
            if i not in temp_gene1:
                child_route1.append(i)
                child_p1 += 1

        for i in route1:
            if child_p2 == index1:
                child_route2.extend(temp_gene2)
                child_p2 += 1
            if i not in temp_gene2:
                child_route2.append(i)
                child_p2 += 1

        pop1 = Pop(np.array(child_route1), chrom1.breakp)
        pop2 = Pop(np.array(child_route2), chrom2.breakp)
        return pop1, pop2

    def cross_pmx(self, chrom1, chrom2):
        # 部分匹配交叉
        route1 = chrom1.route.tolist()
        route2 = chrom2.route.tolist()

        index1, index2 = random.randint(0, len(route1) - 1), random.randint(
            0, len(route1) - 1
        )
        if index1 > index2:
            index1, index2 = index2, index1

        parent_part1, parent_part2 = route1[index1:index2], route2[index1:index2]

        child_route1, child_route2 = [], []
        child_p1, child_p2 = 0, 0

        for i in route1:
            # 指针到达父代的选中部分
            if index1 <= child_p1 < index2:
                # 将父代2选中基因片段复制到子代1指定位置上
                child_route1.append(parent_part2[child_p1 - index1])
                child_p1 += 1
                continue
            if child_p1 < index1 or child_p1 >= index2:
                if i in parent_part2:
                    tmp = parent_part1[parent_part2.index(i)]
                    while tmp in parent_part2:
                        tmp = parent_part1[parent_part2.index(tmp)]
                    child_route1.append(tmp)
                elif i not in parent_part2:
                    child_route1.append(i)
                child_p1 += 1

        for i in route2:
            # 指针到达父代的选中部分
            if index1 <= child_p2 < index2:
                child_route2.append(parent_part1[child_p2 - index1])
                child_p2 += 1
                continue
            # 指针未到达父代的选中部分
            if child_p2 < index1 or child_p2 >= index2:
                # 父代2未选中部分含有父代1选中部分基因
                if i in parent_part1:
                    tmp = parent_part2[parent_part1.index(i)]
                    while tmp in parent_part1:
                        tmp = parent_part2[parent_part1.index(tmp)]
                    child_route2.append(tmp)
                elif i not in parent_part1:
                    child_route2.append(i)
                child_p2 += 1

        pop1 = Pop(np.array(child_route1), chrom1.breakp)
        pop2 = Pop(np.array(child_route2), chrom2.breakp)
        return pop1, pop2

    def cross(self, best_index):
        # 交叉操作
        best_pop = self._population[best_index]

        new_pop = np.empty(self._population_size, dtype=object)
        new_break = np.empty((self._population_size, self._salesman_num - 1), int)
        new_route = np.empty((self._population_size, self._map.target), int)

        i = 0
        while i < self._population_size:
            if i + 1 >= self._population_size:
                # 如果超index范围（大概率不可能超），随机选一个
                new_pop[i] = random.choices(best_pop, k=1)[0]

            two_chrom = random.choices(best_pop, k=2)
            prob = random.random()

            if prob <= self.cross_ox_prob:
                child_route1, child_route2 = self.cross_ox(two_chrom[0], two_chrom[1])
                new_pop[i] = child_route1
                new_pop[i + 1] = child_route2
                i = i + 2
            else:
                child_route1, child_route2 = self.cross_pmx(two_chrom[0], two_chrom[1])
                new_pop[i] = child_route1
                new_pop[i + 1] = child_route2
                i = i + 2

        for j in range(self._population_size):
            # 这一句理论上不要的，要改得删掉全局的route和break，摆了
            new_route[j] = new_pop[j].route
            new_break[j] = new_pop[j].breakp

        self._population = new_pop
        self._population_route = new_route
        self._population_break = new_break
        return

    def mutate_route(self):
        mutate_sum = ["swap", "insert", "reverse"]
        for i in range(self._population_size):
            route = self._population[i].route.tolist()
            index_range = len(route) - 1
            choice = random.choices(mutate_sum, k=1)[0]

            if choice == "swap":
                index1 = random.randint(0, index_range)
                index2 = random.randint(0, index_range)
                route[index1], route[index2] = route[index2], route[index1]
            elif choice == "insert":
                index1, index2 = random.randint(0, index_range), random.randint(
                    0, index_range
                )
                tmp = route.pop(index2)
                route.insert(index1 + 1, tmp)
            elif choice == "reverse":
                index1, index2 = random.randint(0, index_range), random.randint(
                    0, index_range
                )
                if index1 > index2:
                    index1, index2 = index2, index1
                route[index1:index2] = route[index1:index2][::-1]

            self._population[i] = Pop(np.array(route), self._population[i].breakp)
            self._population_route[i] = route

    def mutate_break(self, num):
        # 这里需要考虑到break中数量的变异，但是摆了
        for i in range(self._population_size):
            prob = random.random()
            if prob < self.mutate_break_rate_pop:
                breaks = self.randbreak(
                    self._population[i].route.shape[0], self._salesman_num - 1
                )
                self._population[i] = Pop(self._population[i].route, breaks)
        return

    def check_valid(self, route, breaks):
        ranges = self.get_range(route, breaks)
        best_depots = self.get_best_depot(route, breaks)

        for i, rang in enumerate(ranges):
            cost_onesaleman = 0
            for j in range(len(rang) - 1):
                cost_onesaleman += self._map.get_t2t_distance(rang[j], rang[j + 1])
            cost_onesaleman += self._map.get_d2t_distance(best_depots[i], rang[0])
            cost_onesaleman += self._map.get_d2t_distance(best_depots[i], rang[-1])
            if cost_onesaleman > MAX_DISTANCE:
                return False
        return True

    def invalid_route(self):
        for i in range(self._population_size):
            route = self._population[i].route
            breaks = self._population[i].breakp
            while not self.check_valid(route, breaks):
                # 重新生成一个route和break
                route = np.random.permutation(self._map.target)
                breaks = self.randbreak(route.shape[0], self._salesman_num - 1)
            self._population[i] = Pop(route, breaks)
            self._population_route[i] = route
            self._population_break[i] = breaks

    def ga_process_iterator(self):
        best_pop = Pop(np.array([]), np.array([]))
        best_distance = MAX

        while self.iter < self.num_iter:
            start_time = time.time()

            self.invalid_route()
            # 这里可以考虑将无效个体的判断放在选取最优个体处，可以节省大量开销
            # 但把无效个体判断放前面可以增加种群里的多样性（无效个体重新生成）
            index = self.get_best_chrom(num=8)

            p_cross = random.random()
            if p_cross <= self.cross_rate:
                self.cross(index)

            p_mutate = random.random()
            if p_mutate <= self.mutate_rate:
                self.mutate_route()

            p_mutate_break = random.random()
            if p_mutate_break <= self.mutate_rate:
                self.mutate_break(self._salesman_num - 1)

            best_index = np.argmin(self.get_cost())
            if self.total_distance[best_index] < best_distance:
                best_pop = self._population[best_index]
                best_distance = self.total_distance[best_index]

            self.cost_list.append(self.total_distance[best_index])
            end_time = time.time()

            print(
                f"Epoch {self.iter}\t{self.total_distance[best_index]}\t{round(end_time - start_time,2)}"
            )

            self.iter += 1

        return best_pop


if __name__ == "__main__":
    map = Map()
    ga = GA(64, map, [Order() for _ in range(10)])
    ga.init_population()
    ga.ga_process_iterator()
