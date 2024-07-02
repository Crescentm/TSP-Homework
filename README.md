## 利用遗传算法解决 MDTSP 问题

### 题目要求

无人机配送路径规划问题

无人机可以快速解决最后 10 公里的配送，本作业要求设计一个算法，实现如下图所示区域的无人机配送的路径规划。在此区域中，共有 j 个配送中心，任意一个配送中心有用户所需要的商品，其数量无限，同时任一配送中心的无人机数量无限。该区域同时有 k 个卸货点（无人机只需要将货物放到相应的卸货点即可），假设每个卸货点会随机生成订单，一个订单只有一个商品，但这些订单有优先级别，分为三个优先级别（用户下订单时，会选择优先级别，优先级别高的付费高）：

- 一般：3 小时内配送到即可；

- 较紧急：1.5 小时内配送到；

- 紧急：0.5 小时内配送到。

我们将时间离散化，也就是每隔 t 分钟，所有的卸货点会生成订单（0-m 个订单），同时每隔 t 分钟，系统要做成决策，包括：

1. 哪些配送中心出动多少无人机完成哪些订单；
2. 每个无人机的路径规划，即先完成那个订单，再完成哪个订单，...，最后返回原来的配送中心；
   注意：系统做决策时，可以不对当前的某些订单进行配送，因为当前某些订单可能紧急程度不高，可以累积后和后面的订单一起配送。

目标：一段时间内（如一天），所有无人机的总配送路径最短
约束条件：满足订单的优先级别要求

假设条件：

1. 无人机一次最多只能携带 n 个物品；
2. 无人机一次飞行最远路程为 20 公里（无人机送完货后需要返回配送点）；
3. 无人机的速度为 60 公里/小时；
4. 配送中心的无人机数量无限；
5. 任意一个配送中心都能满足用户的订货需求；

### 解决思路

利用遗传算法强行解决出最优解，更多设置可以参考实验报告

运行示例在[main.ipynb](./main.ipynb)中

遗传算法参照 [MDMTSPV_GA - Multiple Depot Multiple Traveling Salesmen Problem solved by Genetic Algorithm](https://ww2.mathworks.cn/matlabcentral/fileexchange/31814-mdmtspv_ga-multiple-depot-multiple-traveling-salesmen-problem-solved-by-genetic-algorithm)

## Solving the MDTSP Problem Using Genetic Algorithms

### Problem Requirements

Drone Delivery Route Planning Problem

Drones can quickly solve the last 10 kilometers of delivery. This assignment requires designing an algorithm to plan the delivery routes for drones in a given area as shown in the figure below. In this area, there are j delivery centers, any of which can provide the products needed by users in unlimited quantities. Additionally, the number of drones available at any delivery center is also unlimited. The area also contains k drop-off points (drones only need to drop off goods at the corresponding drop-off points). Assume that each drop-off point generates orders randomly, with each order consisting of only one product. These orders have priority levels and are divided into three categories:

- Normal: delivery within 3 hours;
- Semi-Urgent: delivery within 1.5 hours;
- Urgent: delivery within 0.5 hours.

We will discretize time, meaning that every t minutes, all drop-off points generate orders (0 to m orders). Additionally, every t minutes, the system must make decisions, including:

1. Which delivery centers dispatch how many drones to fulfill which orders;
2. The path planning for each drone, i.e., which order to complete first, which order to complete next, and finally return to the original delivery center.

Note: The system can choose not to deliver certain current orders during decision-making if the orders are not urgent and can be accumulated and delivered together with future orders.

Objective: Minimize the total delivery path of all drones over a period of time (e.g., one day)
Constraints: Meet the priority requirements of the orders

Assumptions:

1. A drone can carry at most n items at a time;
2. A drone can fly a maximum distance of 20 kilometers per trip (the drone needs to return to the delivery point after delivering the goods);
3. The speed of the drone is 60 kilometers per hour;
4. The number of drones at any delivery center is unlimited;
5. Any delivery center can meet the user's order demand;

### Solution Approach

Use genetic algorithms to forcefully solve the optimal solution. More settings can be referenced in the experimental report.

You can run my code by using [main.ipynb](./main.ipynb)

Refer to the genetic algorithm at [MDMTSPV_GA - Multiple Depot Multiple Traveling Salesmen Problem solved by Genetic Algorithm](https://ww2.mathworks.cn/matlabcentral/fileexchange/31814-mdmtspv_ga-multiple-depot-multiple-traveling-salesmen-problem-solved-by-genetic-algorithm).
