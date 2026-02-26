import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
import heapq


class YadroSegmentation:
    """
    Tasvir klasterlash (segmentatsiya) jarayonini yakka klass ichida
    OOP tamoyillariga asoslangan holda amalga oshiruvchi klass.
    """

    def __init__(self, data, epsilon=0.8):
        """
        :param data: Masalan, 2D masofalar uchun (N x 2) o'lchamdagi numpy array
        :param epsilon: Chek vazni (threshold) grafiga qirra qo'shishda ishlatiladi
        """
        self.data = data
        self.epsilon = epsilon
        self.G = None            # Yaratiladigan graf
        self.Dt_sequence = None  # Zichlik o'zgarishlar ketma-ketligi
        self.Mt_sequence = None  # Eng zaif tugunlar ketma-ketligi
        self.labels_ = None  # To store the cluster labels after fitting

    def create_graph_with_weights(self):
        """
        data da berilgan nuqtalar orasidagi evklid masofa asosida graph yaratish
        va (max_distance - distance)/max_distance formulasiga o'xshash tarzda vazn berish.
        """
        self.G = nx.Graph()
        
        # Masofalarni hisoblaymiz
        distance_matrix = pdist(self.data, metric='euclidean')
        A = squareform(distance_matrix)
        
        # Max har bir qator bo'yicha (har bir nuqta bo'yicha)
        max_vals = np.max(A, axis=1, keepdims=True)
        weight_matrix = (max_vals - A) / max_vals
        
        # Qirralarni qo‘shish
        n = len(self.data)
        for i in range(n):
            for j in range(i + 1, n):
                if weight_matrix[i, j] > self.epsilon:
                    self.G.add_edge(i, j, weight=weight_matrix[i, j])

    def find_max_min_neighbors(self):
        """
        Grafiga asoslangan holda maksimal va minimal darajali tugunlarni topish.
        Izolyatsiyalangan tugunlar ro‘yxatini ham qaytaradi.
        """
        if self.G is None:
            raise ValueError("Graf hali yaratilmagan. create_graph_with_weights() chaqiring.")

        degrees = dict(self.G.degree())
        max_degree_node = max(degrees, key=degrees.get)
        min_degree_node = min(degrees, key=degrees.get)
        max_degree = degrees[max_degree_node]
        min_degree = degrees[min_degree_node]

        isolated_nodes = [node for node, deg in self.G.degree() if deg == 0]

        return max_degree_node, max_degree, min_degree_node, min_degree, isolated_nodes

    def compute_local_density(self, G):
        """
        Tugunlarning qo‘shnilari bilan qirra vaznlarini yig‘indi sifatida
        lokal zichlik (density) qiymatini qaytaradi.
        """
        density = {}
        for node in G.nodes:
            density[node] = sum(G[node][nbr]['weight'] for nbr in G.neighbors(node))
        return density

    def compute_density_variation_sequence(self):
        """
        Zichlik o'zgarishlar ketma-ketligini (Dt_sequence) va eng zaif tugunlar ketma-ketligini (Mt_sequence)
        heapq yordamida hisoblaydi.
        """
        if self.G is None:
            raise ValueError("Graf hali yaratilmagan. create_graph_with_weights() chaqiring.")

        # Lokal zichliklarni boshlang‘ich hisoblash
        density = self.compute_local_density(self.G)
        H = list(self.G.nodes)
        heap = [(density[node], node) for node in H]
        heapq.heapify(heap)

        Dt_sequence = []
        Mt_sequence = []
        # G ni nusxa olamiz, chunki pop qilingan node larni olib tashlash kerak
        G_copy = self.G.copy()

        while heap:
            Dt, Mt = heapq.heappop(heap)
            Dt_sequence.append(Dt)
            Mt_sequence.append(Mt)

            # Qo‘shnilar zichligini yangilash
            neighbors = list(G_copy.neighbors(Mt))
            for nbr in neighbors:
                density[nbr] -= G_copy[Mt][nbr]['weight']

            # Tugunni olib tashlaymiz
            G_copy.remove_node(Mt)

            # Heapni qayta qurish
            heap = [(density[node], node) for node in G_copy.nodes]
            heapq.heapify(heap)

        self.Dt_sequence = Dt_sequence
        self.Mt_sequence = Mt_sequence

        return Dt_sequence, Mt_sequence

    def identify_core_pixels(self, Dt_sequence, Mt_sequence, delta, beta):
        """
        Zichlik pasayish tezligi R_t ga asoslangan holda asosiy (core) tugunlarni aniqlash.
        :param Dt_sequence: Zichliklar ketma-ketligi
        :param Mt_sequence: Eng zaif tugunlar ketma-ketligi
        :param delta: R_t ni saralashdagi nisbiy threshold (masalan, 0.5)
        :param beta: Ketma-ket R_t > alpha bo'lgan minimal son (masalan, 5)
        :return: core_pixels ro‘yxati
        """
        # R_t = (D_t - D_(t+1)) / D_t
        Rt_sequence = []
        for t in range(len(Dt_sequence) - 1):
            if Dt_sequence[t] != 0:
                Rt_sequence.append((Dt_sequence[t] - Dt_sequence[t + 1]) / Dt_sequence[t])
            else:
                Rt_sequence.append(0)

        # Ijobiy R_t larni saralab, sorted qilamiz
        positive_Rt = [r for r in Rt_sequence if r > 0]
        if not positive_Rt:
            return []

        positive_Rt_sorted = sorted(positive_Rt)
        alpha_index = int(len(positive_Rt_sorted) * delta)
        if alpha_index >= len(positive_Rt_sorted):
            alpha_index = len(positive_Rt_sorted) - 1

        alpha = positive_Rt_sorted[alpha_index]

        # Asosiy piksellar to'plamini aniqlash
        core_pixels = []
        consecutive_count = 0

        for t in range(len(Rt_sequence)):
            if Rt_sequence[t] > alpha:
                consecutive_count += 1
                if consecutive_count >= beta:
                    core_pixels.append(Mt_sequence[t])
            else:
                consecutive_count = 0

        return core_pixels

    def partition_core_pixels(self, G, core_pixels, theta=0.1):
        """
        Asosiy tugunlardan subgraf (core_graph) tuzib, past vaznli qirralar (w < theta) ni olib tashlaydi
        va ulanmagan komponentlarni topadi.
        """
        core_graph = G.subgraph(core_pixels).copy()
        # Zaif qirralarni olib tashlash
        weak_edges = [(u, v) for u, v, w in core_graph.edges(data='weight') if w < theta]
        core_graph.remove_edges_from(weak_edges)

        # Ulanmagan komponentlarni (connected components) topish
        segments = list(nx.connected_components(core_graph))
        return segments

    def calculate_similarity(self, G, pixel, segment):
        """
        Berilgan tugun (pixel) va segment orasidagi o‘rtacha vazn (similarity) ni hisoblash.
        """
        weights = [G[pixel][s]['weight'] for s in segment if G.has_edge(pixel, s)]
        if weights:
            return np.mean(weights)
        return 0

    def expand_segments(self, G, Mt_sequence, core_segments, lambda_value=0.5):
        """
        Orqaga ketma-ketlikda (Mt_sequence) tugunlarni klasterlarga qo‘shish (segmentlarni kengaytirish).
        :return: (segments, low_confidence_pixels)
        """
        segments = [list(s) for s in core_segments]
        low_confidence_pixels = set()
        added_pixels = set()

        n = len(Mt_sequence)
        # Orqadan oldinga qarab (t = n-1 -> 0)
        for t in range(n - 1, -1, -1):
            Mt = Mt_sequence[t]

            if Mt not in added_pixels:
                # Har bir segment bilan o‘xshashlikni o‘lchaymiz
                similarities = []
                for i, seg in enumerate(segments):
                    sim_val = self.calculate_similarity(G, Mt, seg)
                    if sim_val > 0:
                        similarities.append((i, sim_val))

                if similarities:
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    s1, m1 = similarities[0]
                    m2 = 0
                    if len(similarities) > 1:
                        m2 = similarities[1][1]

                    # Eng o‘xshash segmentga qo‘shish
                    if m1 > 0:
                        segments[s1].append(Mt)
                        added_pixels.add(Mt)
                        if m2 > lambda_value * m1:
                            low_confidence_pixels.add(Mt)
                else:
                    # Hech bir segmentga mos kelmasa – yangi segment yaratamiz
                    segments.append([Mt])
                    added_pixels.add(Mt)

        # Har bir segmentdan takrorlanayotgan tugunlarni olib tashlaymiz
        for i in range(len(segments)):
            segments[i] = list(set(segments[i]))

        # Barcha tugunlar qo‘shildimi?
        all_nodes = set(G.nodes)
        added_nodes = set().union(*segments)
        missing_nodes = all_nodes - added_nodes

        # Qo'shilmay qolgan tugunlar bo'lsa, alohida segment sifatida qo'shamiz
        for node in missing_nodes:
            print(f"Tugun {node} segmentga qo'shilmagan, yangi segmentga ajratiladi.")
            segments.append([node])

        return segments, low_confidence_pixels

    def compute_DVI(self, G, clusters):
        """
        Klasterlar uchun 3 xil DVI qiymatlarini (DVI1, DVI2, DVI3) hisoblash.
        """
        node_to_cluster = {}
        for i, cluster in enumerate(clusters):
            for node in cluster:
                node_to_cluster[node] = i

        intra_density = {node: 0 for node in G.nodes()}
        inter_density = {node: 0 for node in G.nodes()}

        for u, v, data in G.edges(data=True):
            w = data.get('weight', 1)
            if node_to_cluster[u] == node_to_cluster[v]:
                intra_density[u] += w
                intra_density[v] += w
            else:
                inter_density[u] += w
                inter_density[v] += w

        # DVI1
        dvi1_value = sum(intra_density.values()) - sum(inter_density.values())

        # DVI2 = Inter / Intra
        intra_sum = sum(intra_density.values())
        inter_sum = sum(inter_density.values())
        if intra_sum == 0:
            dvi2_value = float('inf')
        else:
            dvi2_value = inter_sum / intra_sum

        # DVI3
        dvi3_value = 0
        for node in G.nodes():
            if intra_density[node] > 0:
                dvi3_value += (inter_density[node] / intra_density[node])

        return dvi1_value, dvi2_value, dvi3_value

    def visualize_clusters(self, segments, show=True):
        """
        Klasterlash natijalarini 2D ko‘rinishda tasvirlab berish.
        self.data[x] = 2D koordinata
        segments - klaster bo‘lib ajratilgan tugun ro‘yxatlari
        """
        # Pozitsiya (nuqta) ni data dagi koordinatalarga tayansak bo'ladi
        pos = {i: self.data[i] for i in range(len(self.data))}
        color_list = ['r', 'g', 'b', 'y', 'c', 'm']  # ixtiyoriy ranglar ro‘yxati

        for idx, cluster in enumerate(segments):
            nx.draw_networkx_nodes(
                self.G, pos,
                nodelist=cluster,
                node_color=color_list[idx % len(color_list)],
                node_size=25
            )
        nx.draw_networkx_edges(self.G, pos, alpha=0.25)

        if show:
            plt.show()

    def full_pipeline(self, delta=0.5, beta=5, theta=0.1, lambda_value=0.5, visualize=True):
        """
        Barcha bosqichlarni (graf yaratish, zichlikni hisoblash, asosiy piksellar,
        segmentlash, DVI) ketma-ket bajaruvchi qulay metod.
        """
        # 1. Graf yaratish
        self.create_graph_with_weights()

        # 2. Zichlik o‘zgarishlar ketma-ketligi
        if self.Dt_sequence is None or self.Mt_sequence is None:
            self.compute_density_variation_sequence()

        # 3. Asosiy piksellar (core) ni aniqlash
        core_pixels = self.identify_core_pixels(
            self.Dt_sequence,
            self.Mt_sequence,
            delta=delta,
            beta=beta
        )

        # 4. Core pixellar asosida bo‘linishni aniqlash
        core_segments = self.partition_core_pixels(self.G, core_pixels, theta=theta)

        # 5. Segmentlarni kengaytirish
        segments, low_confidence_pixels = self.expand_segments(
            self.G,
            self.Mt_sequence,
            core_segments,
            lambda_value=lambda_value
        )

        # 6. DVI hisoblash
        dvi1, dvi2, dvi3 = self.compute_DVI(self.G, segments)

        # 7. Vizual ko‘rinish
        if visualize:
            print(f"Klasterning umumiy soni: {len(segments)}")
            print(f"DVI1 = {dvi1}, DVI2 = {dvi2}, DVI3 = {dvi3}")
            self.visualize_clusters(segments, show=True)

        # Eng katta indeksni topish
        max_index = max(max(class_group) for class_group in segments)
        
        # Sinflarga mos ravishda numpy vektorini yaratish
        classified_array = np.full(max_index + 1, -1)  # Barcha qiymatlarni -1 bilan boshlash
        for class_label, indices in enumerate(segments):
            classified_array[indices] = class_label

        self.labels_ = classified_array
        return self, dvi1, dvi2, dvi3
        

    def find_optimal_parameters(self, delta_values, beta=5):
        """
        delta ning turli qiymatlari bo‘yicha pipeline ishga tushirib,
        DVI3 ga ko‘ra optimal parametrni tanlash (misol sifatida).
        """
        best_params = None
        best_score = 0
        best_class = None

        # Avval graf yaratib, Dt va Mt ni hisoblab olish
        self.create_graph_with_weights()
        self.compute_density_variation_sequence()

        for delta in delta_values:
            # Har safar (grafning yangilangan nusxasi kerak bo‘lishi mumkin), 
            # bunda original G ni saqlab qolish yoki har iteratsiyada copy() dan foydalanish lozim.
            # Soddalik uchun shu yerning o‘zida klonlab ishlash mumkin, ammo demoda shu tariqa qoldiramiz.

            # Original G ni saqlab olamiz
            original_G = self.G.copy()

            # Yangi segmentatsiya
            core_pixels = self.identify_core_pixels(self.Dt_sequence, self.Mt_sequence, delta, beta)
            core_segments = self.partition_core_pixels(self.G, core_pixels, theta=0.1)
            segments, _ = self.expand_segments(self.G, self.Mt_sequence, core_segments, lambda_value=0.5)
            dvi1, dvi2, dvi3 = self.compute_DVI(self.G, segments)

            # Qayta tiklaymiz
            self.G = original_G

            # DVI3 ni minimallashtirishni maqsad qilamiz (yoki vaziyatga qarab boshqa o‘lchov)
            if dvi1 > best_score and dvi1 > 0:
                best_score = dvi1
                best_params = delta
                best_class = len(segments)

        return best_params, best_score, best_class

