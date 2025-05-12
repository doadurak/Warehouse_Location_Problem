import os
import random
import pandas as pd
from pulp import LpProblem, lpSum, LpMinimize, LpVariable, lpSum, LpBinary, value, GUROBI_CMD
DATASET_DIR = "datasets"
OUTPUT_DIR = "outputs"

def get_dataset_files():
    return [f for f in os.listdir(DATASET_DIR) if f.endswith('.txt')]

def read_input_file(filepath):
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    depot_count, customer_count = map(int, lines[0].split())

    depots = []
    for i in range(1, depot_count + 1):
        cap, cost = map(float, lines[i].split())
        depots.append({'capacity': cap, 'cost': cost, 'remaining': cap})

    customers = []
    index = depot_count + 1
    while index < len(lines):
        demand = float(lines[index])
        index += 1
        costs = list(map(float, lines[index].split()))
        customers.append({'demand': demand, 'costs': costs})
        index += 1

    return depots, customers

def validate_assignment_cost(assignments, depots, customers):
    used_depots = set(assignments)
    depot_capacities = [d['capacity'] for d in depots]
    total_cost = 0

    for c_idx, d_idx in enumerate(assignments):
        demand = customers[c_idx]['demand']
        if depot_capacities[d_idx] < demand:
            raise ValueError(f"Müşteri {c_idx} kapasite aşımıyla {d_idx} numaralı depoya atanmış.")
        depot_capacities[d_idx] -= demand
        total_cost += customers[c_idx]['costs'][d_idx]

    total_cost += sum(depots[d]['cost'] for d in used_depots)
    return total_cost


def best_solver_300(depots, customers, generations=100, pop_size=40, mutation_rate=0.15, time_limit=1800):
    import random
    from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, GUROBI_CMD, value

    num_customers = len(customers)
    num_depots = len(depots)

    def fitness(assign):
        used = set()
        remaining = [d['capacity'] for d in depots]
        cost = 0
        for i, d in enumerate(assign):
            if remaining[d] < customers[i]['demand']:
                return float('inf')
            remaining[d] -= customers[i]['demand']
            used.add(d)
            cost += customers[i]['costs'][d]
        cost += sum(depots[d]['cost'] for d in used)
        return cost

    def generate_individual():
        ind = []
        for c in customers:
            feasible = [i for i, d in enumerate(depots) if d['capacity'] >= c['demand']]
            if not feasible:
                return None
            ind.append(random.choice(feasible))
        return ind

    def crossover(p1, p2):
        cut = random.randint(0, num_customers - 1)
        return p1[:cut] + p2[cut:]

    def mutate(ind):
        new = ind[:]
        for i in range(num_customers):
            if random.random() < mutation_rate:
                new[i] = random.randint(0, num_depots - 1)
        return new

    population = []
    while len(population) < pop_size:
        ind = generate_individual()
        if ind and fitness(ind) < float('inf'):
            population.append(ind)

    for _ in range(generations):
        population.sort(key=fitness)
        next_gen = population[:10]  
        while len(next_gen) < pop_size:
            p1, p2 = random.sample(population[:15], 2)
            child = crossover(p1, p2)
            child = mutate(child)
            if fitness(child) < float('inf'):
                next_gen.append(child)
        population = next_gen

    best = min(population, key=fitness)

    prob = LpProblem("Best_WLP_300", LpMinimize)
    x = [[LpVariable(f"x_{c}_{d}", cat=LpBinary) for d in range(num_depots)] for c in range(num_customers)]
    y = [LpVariable(f"y_{d}", cat=LpBinary) for d in range(num_depots)]

    prob += lpSum(customers[c]['costs'][d] * x[c][d]
                  for c in range(num_customers) for d in range(num_depots)) + \
            lpSum(depots[d]['cost'] * y[d] for d in range(num_depots))

    for c in range(num_customers):
        prob += lpSum(x[c][d] for d in range(num_depots)) == 1
    for d in range(num_depots):
        prob += lpSum(customers[c]['demand'] * x[c][d]
                      for c in range(num_customers)) <= depots[d]['capacity']
    for c in range(num_customers):
        for d in range(num_depots):
            prob += x[c][d] <= y[d]

    for c in range(num_customers):
        for d in range(num_depots):
            x[c][d].setInitialValue(1 if best[c] == d else 0)
    used = set(best)
    for d in range(num_depots):
        y[d].setInitialValue(1 if d in used else 0)

    solver = GUROBI_CMD(msg=True, timeLimit=time_limit)
    prob.solve(solver)

    total_cost = value(prob.objective)
    assignments = []
    for c in range(num_customers):
        for d in range(num_depots):
            if x[c][d].value() == 1:
                assignments.append(d)
                break

    return total_cost, assignments

# def hybrid_greedy_ilp(depots, customers, time_limit=180, greedy_trials=5):
#     from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, value, GUROBI_CMD
#     import random

#     num_customers = len(customers)
#     num_depots = len(depots)

#     def greedy_assignment(customer_order):
#         assignments = []
#         depot_remaining = [depot['capacity'] for depot in depots]
#         depot_opened = [False] * num_depots
#         total_cost = 0

#         for c in customer_order:
#             best_depot = -1
#             best_cost = float('inf')
#             for d in range(num_depots):
#                 if depot_remaining[d] >= customers[c]['demand']:
#                     cost = customers[c]['costs'][d]
#                     if not depot_opened[d]:
#                         cost += depots[d]['cost']
#                     if cost < best_cost:
#                         best_cost = cost
#                         best_depot = d
#             if best_depot == -1:
#                 return None, None, float('inf')  # Geçersiz
#             assignments.append(best_depot)
#             depot_remaining[best_depot] -= customers[c]['demand']
#             depot_opened[best_depot] = True
#             total_cost += customers[c]['costs'][best_depot]
#             if not depot_opened[best_depot]:
#                 total_cost += depots[best_depot]['cost']

#         return assignments, depot_opened, total_cost

#     # 🔁 Multi-start Greedy
#     best_assignments = None
#     best_depot_opened = None
#     best_cost = float('inf')

#     for _ in range(greedy_trials):
#         order = list(range(num_customers))
#         random.shuffle(order)
#         assignments, opened, cost = greedy_assignment(order)
#         if assignments and cost < best_cost:
#             best_assignments = assignments
#             best_depot_opened = opened
#             best_cost = cost

#     if best_assignments is None:
#         raise Exception("Greedy başlangıçta kapasite aşıldı!")

#     # ✅ ILP kısmı
#     prob = LpProblem("Hybrid_WLP", LpMinimize)
#     x = [[LpVariable(f"x_{c}_{d}", cat=LpBinary) for d in range(num_depots)] for c in range(num_customers)]
#     y = [LpVariable(f"y_{d}", cat=LpBinary) for d in range(num_depots)]

#     prob += lpSum(customers[c]['costs'][d] * x[c][d] for c in range(num_customers) for d in range(num_depots)) + \
#             lpSum(depots[d]['cost'] * y[d] for d in range(num_depots))

#     for c in range(num_customers):
#         prob += lpSum(x[c][d] for d in range(num_depots)) == 1

#     for d in range(num_depots):
#         prob += lpSum(customers[c]['demand'] * x[c][d] for c in range(num_customers)) <= depots[d]['capacity']

#     for c in range(num_customers):
#         for d in range(num_depots):
#             prob += x[c][d] <= y[d]

#     # 🔥 Warm Start
#     for c in range(num_customers):
#         for d in range(num_depots):
#             if d == best_assignments[c]:
#                 x[c][d].setInitialValue(1)
#                 x[c][d].fixValue()
#             else:
#                 x[c][d].setInitialValue(0)

#     for d in range(num_depots):
#         y[d].setInitialValue(1 if best_depot_opened[d] else 0)

#     # 🚀 Solve
#     solver = GUROBI_CMD(msg=True, timeLimit=time_limit, options=[("MIPGap", 0.01), ("Threads", 4)])
#     prob.solve(solver)

#     # ✅ Sonuçları çek
#     assignments = []
#     for c in range(num_customers):
#         for d in range(num_depots):
#             if x[c][d].value() == 1:
#                 assignments.append(d)
#                 break

#     total_cost = value(prob.objective)
#     return total_cost, assignments

def hybrid_greedy_ilp(depots, customers, time_limit=180):

    num_customers = len(customers)
    num_depots = len(depots)

    greedy_assignments = []
    depot_remaining = [depot['capacity'] for depot in depots]
    depot_opened = [False] * num_depots

    customer_indices = list(range(num_customers))
    random.shuffle(customer_indices) 

    for c in customer_indices:
        best_depot = -1
        best_score = float('inf')
        for d in range(num_depots):
            if depot_remaining[d] >= customers[c]['demand']:
                cost = customers[c]['costs'][d]
                fixed_cost = depots[d]['cost'] if not depot_opened[d] else 0
                score = (cost + fixed_cost) / (depot_remaining[d] + 1e-6)
                if score < best_score:
                    best_score = score
                    best_depot = d
        if best_depot == -1:
            raise Exception("Greedy başlangıçta kapasite aşıldı!")
        greedy_assignments.append(best_depot)
        depot_remaining[best_depot] -= customers[c]['demand']
        depot_opened[best_depot] = True

    prob = LpProblem("Hybrid_WLP", LpMinimize)
    x = [[None for _ in range(num_depots)] for _ in range(num_customers)]
    y = [LpVariable(f"y_{d}", cat=LpBinary) for d in range(num_depots)]

    for c in range(num_customers):
        for d in range(num_depots):
            if customers[c]['demand'] <= depots[d]['capacity']:
                x[c][d] = LpVariable(f"x_{c}_{d}", cat=LpBinary)

    prob += lpSum(
        customers[c]['costs'][d] * x[c][d]
        for c in range(num_customers)
        for d in range(num_depots)
        if x[c][d] is not None
    ) + lpSum(depots[d]['cost'] * y[d] for d in range(num_depots))

    for c in range(num_customers):
        prob += lpSum(
            x[c][d] for d in range(num_depots) if x[c][d] is not None
        ) == 1

    for d in range(num_depots):
        prob += lpSum(
            customers[c]['demand'] * x[c][d]
            for c in range(num_customers)
            if x[c][d] is not None
        ) <= depots[d]['capacity']

    for c in range(num_customers):
        for d in range(num_depots):
            if x[c][d] is not None:
                prob += x[c][d] <= y[d]
    for c in range(num_customers):
        for d in range(num_depots):
            if x[c][d] is not None:
                x[c][d].setInitialValue(1 if d == greedy_assignments[c] else 0)

    for d in range(num_depots):
        y[d].setInitialValue(1 if depot_opened[d] else 0)

    solver = GUROBI_CMD(
        msg=1,
        timeLimit=time_limit,
        options=[
            ("Threads", 4),
            ("MIPFocus", 1),
            ("Heuristics", 0.5),
            ("Presolve", 2)
        ]
    )
    prob.solve(solver)
    assignments = []
    for c in range(num_customers):
        assigned = False
        for d in range(num_depots):
            if x[c][d] is not None and x[c][d].value() == 1:
                assignments.append(d)
                assigned = True
                break
        if not assigned:
            assignments.append(-1)  

    total_cost = value(prob.objective)
    return total_cost, assignments


def ilp_solver(depots, customers):
    prob = LpProblem("WLP", LpMinimize)

    num_customers = len(customers)
    num_depots = len(depots)

    x = [[LpVariable(f"x_{c}_{d}", cat=LpBinary) for d in range(num_depots)] for c in range(num_customers)]
    y = [LpVariable(f"y_{d}", cat=LpBinary) for d in range(num_depots)]

    prob += lpSum(customers[c]['costs'][d] * x[c][d] for c in range(num_customers) for d in range(num_depots)) + \
            lpSum(depots[d]['cost'] * y[d] for d in range(num_depots))

    for c in range(num_customers):
        prob += lpSum(x[c][d] for d in range(num_depots)) == 1

    for d in range(num_depots):
        prob += lpSum(customers[c]['demand'] * x[c][d] for c in range(num_customers)) <= depots[d]['capacity']

    for c in range(num_customers):
        for d in range(num_depots):
            prob += x[c][d] <= y[d]

    solver = GUROBI_CMD(msg=True, timeLimit=600)
    prob.solve(solver)
    assignments = []
    total_cost = prob.objective.value()

    for c in range(num_customers):
        for d in range(num_depots):
            if x[c][d].value() == 1:
                assignments.append(d)
                break

    return total_cost, assignments


def strongest_hybrid_solver(depots, customers, time_limit=3600):
    from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, GUROBI_CMD, value
    import random

    num_customers = len(customers)
    num_depots = len(depots)

    def fitness(assign):
        used = set()
        rem = [d['capacity'] for d in depots]
        cost = 0
        for i, d in enumerate(assign):
            if rem[d] < customers[i]['demand']:
                return float('inf')
            rem[d] -= customers[i]['demand']
            used.add(d)
            cost += customers[i]['costs'][d]
        cost += sum(depots[d]['cost'] for d in used)
        return cost

    def create_individual():
        ind = []
        for c in customers:
            feasible = [i for i, d in enumerate(depots) if d['capacity'] >= c['demand']]
            if not feasible:
                return None
            ind.append(random.choice(feasible))
        return ind

    def mutate(ind, mutation_rate=0.05):
        new = ind[:]
        for i in range(num_customers):
            if random.random() < mutation_rate:
                new[i] = random.randint(0, num_depots - 1)
        return new

    def crossover(p1, p2):
        k = random.randint(0, num_customers - 1)
        return p1[:k] + p2[k:]

    population = []
    population_size = 100
    generations = 200

    while len(population) < population_size:
        ind = create_individual()
        if ind and fitness(ind) < float('inf'):
            population.append(ind)

    for _ in range(generations):
        population.sort(key=fitness)
        next_gen = population[:10]  

        while len(next_gen) < population_size:
            p1, p2 = random.sample(population[:30], 2)
            child = crossover(p1, p2)
            child = mutate(child)
            if fitness(child) < float('inf'):
                next_gen.append(child)
        population = next_gen

    best = min(population, key=fitness)

    prob = LpProblem("Strongest_WLP", LpMinimize)
    x = [[LpVariable(f"x_{c}_{d}", cat=LpBinary) for d in range(num_depots)] for c in range(num_customers)]
    y = [LpVariable(f"y_{d}", cat=LpBinary) for d in range(num_depots)]

    prob += (
        lpSum(customers[c]['costs'][d] * x[c][d]
              for c in range(num_customers) for d in range(num_depots)) +
        lpSum(depots[d]['cost'] * y[d] for d in range(num_depots))
    )

    for c in range(num_customers):
        prob += lpSum(x[c][d] for d in range(num_depots)) == 1
    for d in range(num_depots):
        prob += lpSum(customers[c]['demand'] * x[c][d] for c in range(num_customers)) <= depots[d]['capacity']
    for c in range(num_customers):
        for d in range(num_depots):
            prob += x[c][d] <= y[d]
    for c in range(num_customers):
        for d in range(num_depots):
            x[c][d].setInitialValue(1 if best[c] == d else 0)

    used_depots = set(best)
    for d in range(num_depots):
        y[d].setInitialValue(1 if d in used_depots else 0)

    solver = GUROBI_CMD(msg=True, timeLimit=time_limit)
    prob.solve(solver)

    total_cost = value(prob.objective)
    assignments = []
    for c in range(num_customers):
        for d in range(num_depots):
            if x[c][d].value() == 1:
                assignments.append(d)
                break

    return total_cost, assignments

def main():
    dataset_files = get_dataset_files()
    results = []

    for file_name in dataset_files:
        print(f"🟢 {file_name} işleniyor...")
        path = os.path.join(DATASET_DIR, file_name)

        try:
            depots, customers = read_input_file(path)

            if "25" in file_name:
                print(" En güçlü hibrit çözüm (Genetik + CBC)...")
                total_cost, assignments = strongest_hybrid_solver(depots, customers, time_limit=1800)

            elif any(keyword in file_name for keyword in ["50", "200"]):
                 print(" ILP çözüm kullanılıyor...")
                 total_cost, assignments = ilp_solver(depots, customers)

            elif "300" in file_name:
                 print(" 300 için Best Hybrid ILP çözümü uygulanıyor...")
                 total_cost, assignments = best_solver_300(depots, customers)


            elif file_name == "wl_500.txt":
                print("500 için Hybrid çözüm (geliştirilmiş) uygulanıyor...")
                total_cost, assignments = hybrid_greedy_ilp(
                depots,
                customers,
                time_limit=300 
    )


            else:
                raise Exception("Bilinmeyen dosya tipi.")

            true_cost = validate_assignment_cost(assignments, depots, customers)
            print(f" Gurobi maliyeti: {total_cost:.3f}")
            print(f" Gerçek maliyet (doğrulama): {true_cost:.3f}")

            if abs(true_cost - total_cost) > 1e-2:
                raise ValueError(f"UYARI: Maliyet uyuşmazlığı! Gurobi: {total_cost:.3f}, Gerçek: {true_cost:.3f}")

            results.append({
                'Dosya Adı': file_name,
                'Müşteri Sayısı': len(customers),
                'Toplam Maliyet': round(total_cost, 3),
                'Atamalar': ' '.join(map(str, assignments))
            })

        except Exception as e:
            print(f" {file_name} işlenirken hata oluştu:", str(e))
            results.append({
                'Dosya Adı': file_name,
                'Müşteri Sayısı': 'HATA',
                'Toplam Maliyet': 'HATA',
                'Atamalar': str(e)
            })

    # Excel çıktısı
    df = pd.DataFrame(results)
    excel_path = os.path.join(OUTPUT_DIR, "wlp_sonuclar.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"📄 Sonuçlar başarıyla {excel_path} dosyasına yazıldı.")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()

