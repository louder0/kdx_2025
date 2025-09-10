import math
from collections import deque

from coords_file import is_point_on_line


# ------------------------task-0------------------------


def predict_qq_with_yolo(cut_imgs, model, qq_txt_path):
    circles = []
    for pngxy, img in cut_imgs:
        results = model(img)
        pngx, pngy = pngxy
        for result in results[0].boxes:
            if result.conf.cpu().tolist()[0] < 0.7:
                continue
            x0, y0, x1, y1 = result.xyxy.cpu().tolist()[0]
            x0 += pngx
            x1 += pngx
            y0 += pngy
            y1 += pngy
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            circles.append((cx, cy))

    with open(qq_txt_path, "w", encoding="utf-8") as txt:
        for circle in circles:
            txt.write(f"{circle}\n")

    return circles


# ------------------------task-1------------------------


def is_wanted_line(draw, features):
    for feature in features:
        if feature not in draw:
            print(f"dict denied: {feature} not in dict")
            return False

    # items在get_drawings()里面测 把所有'l'揉到一起
    if draw["type"] != "s" and draw["type"] != "fs":
        # print(f'dict denied: type is {draw["type"]}')
        return False  # 绘图类型 's'代表描边
        # if draw['color'] != (0.0, 0.0, 0.0):
        # print(f'dict denied: color is {draw["color"]}')
        return False  # 描边颜色 rgb 0-1
    if draw["stroke_opacity"] != 1.0:  # 给qq6服务
        # print(f'dict denied: stroke_opacity is {draw["stroke_opacity"]}')
        return False  # 描边透明度 0.0完全透明 - 1.0完全不透明
    if "Hatch" in draw.get("layer", ""):
        return False

    rect = draw["rect"]
    if abs(rect.x0 - rect.x1) + abs(rect.y0 - rect.y1) < 2:
        # print('dict denied: too small')
        return False

    return True
    """
    {'items': 
        [('l', Point(1061.0399169921875, 508.47998046875), Point(1060.919921875, 508.3599853515625)), 
        ('l', Point(1060.919921875, 508.3599853515625), Point(1060.7999267578125, 508.239990234375)), 
        ('l', Point(1060.7999267578125, 508.239990234375), Point(1060.679931640625, 508.3599853515625)), 
        ('l', Point(1060.679931640625, 508.3599853515625), Point(1060.5599365234375, 508.47998046875)), 
        ('l', Point(1060.5599365234375, 508.47998046875), Point(1060.679931640625, 508.5999755859375)), 
        ('l', Point(1060.679931640625, 508.5999755859375),Point(1060.7999267578125, 508.719970703125)), 
        ('l', Point(1060.7999267578125, 508.719970703125), Point(1060.919921875, 508.5999755859375)), 
        ('l', Point(1060.919921875, 508.5999755859375), Point(1061.0399169921875, 508.47998046875))], 
    'closePath': False, 
    'type': 'fs', 
    'even_odd': False, 
    'fill_opacity': 1.0, 
    'fill': (0.0, 0.0, 0.0), 
    'rect': Rect(1060.5599365234375, 508.239990234375, 1061.0399169921875, 508.719970703125), 
    'seqno': 10393, 
    'layer': 
    'x-GND|Hatch Over', 
    'stroke_opacity': 1.0, 
    'color': (0.0, 0.0, 0.0), 
    'width': 0.0, 
    'lineCap': (1, 1, 1),
    'lineJoin': 0.11999999731779099, 
    'dashes': '[] 0'
    }
    """


# ------------------------task-2-------------------------


def get_cross_line_angles(p1, p2, p3, p4, tol=2):
    # 向量
    vx, vy = p2[0] - p1[0], p2[1] - p1[1]
    wx, wy = p4[0] - p3[0], p4[1] - p3[1]
    # 模长
    nv = math.hypot(vx, vy)
    nw = math.hypot(wx, wy)
    # 如果任意一条线退化为点，就直接返回 [0,0]
    if nv < tol or nw < tol:
        return [0.0, 0.0]
    # 点积
    dot = vx * wx + vy * wy
    # 防止浮点误差
    cosθ = max(-1.0, min(1.0, dot / (nv * nw)))
    θ = math.degrees(math.acos(cosθ))
    return [θ, 180.0 - θ]


def did_intersect(p1, p2, p3, p4, tol=2):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    # 叉积符号函数
    def ori(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    # 先做正常相交检测（X形、T形、端点接触）
    o1 = ori(x1, y1, x2, y2, x3, y3)
    o2 = ori(x1, y1, x2, y2, x4, y4)
    o3 = ori(x3, y3, x4, y4, x1, y1)
    o4 = ori(x3, y3, x4, y4, x2, y2)
    if o1 * o2 <= 0 and o3 * o4 <= 0:
        return True

    # 否则再检测“近似相交”：四个端点到对方线段的最小距离
    def point_to_seg_dist(px, py, x0, y0, x1, y1):
        vx, vy = x1 - x0, y1 - y0
        wx, wy = px - x0, py - y0
        # 投影系数
        t = (vx * wx + vy * wy) / (vx * vx + vy * vy) if (vx * vx + vy * vy) > 0 else 0
        t = max(0.0, min(1.0, t))
        cx, cy = x0 + t * vx, y0 + t * vy
        return math.hypot(px - cx, py - cy)

    # 端点到线段的最小距离
    dists = [
        point_to_seg_dist(x1, y1, x3, y3, x4, y4),
        point_to_seg_dist(x2, y2, x3, y3, x4, y4),
        point_to_seg_dist(x3, y3, x1, y1, x2, y2),
        point_to_seg_dist(x4, y4, x1, y1, x2, y2),
    ]
    return min(dists) <= tol


def get_gangangs(not_kdx, kdx_lines, lens=((33, 25, 25), 55), tol=1):
    gangangs90 = {}
    gangangs45 = {}  # 两个都可以是的 避免后面if/elif把这类扔掉
    good_gg90_c = 0
    good_gg45_c = 0

    min_len, max_len = lens

    for kx0, ky0, kx1, ky1 in kdx_lines:
        kdx_gg90mid = []
        kdx_gg45 = []
        for gx0, gy0, gx1, gy1 in not_kdx:
            p1, p2, p3, p4 = (kx0, ky0), (kx1, ky1), (gx0, gy0), (gx1, gy1)
            if not did_intersect(p1, p2, p3, p4):
                continue

            a1, a2 = get_cross_line_angles(p1, p2, p3, p4)
            is90d = abs(a1 - 90) < tol and abs(a2 - 90) < tol
            is45d = abs(min(a1, a2) - 45) < tol and abs(max(a1, a2) - 135) < tol
            line_len = math.hypot(abs(gx1 - gx0), abs(gy1 - gy0))
            # min_len依次是kdx_gg90mid, kdx_gg45和kdx_gg90

            if is90d:
                if not max_len > line_len > min_len[2]:
                    continue
                if not is_point_on_line(
                    p1, p2, ((gx1 + gx0) / 2, (gy1 + gy0) / 2), tol=3
                ):
                    continue
                if not line_len > min_len[0]:
                    continue
                kdx_gg90mid.append((gx0, gy0, gx1, gy1))

            elif is45d:
                if not max_len > line_len > min_len[1]:
                    continue
                if not is_point_on_line(
                    p1, p2, ((gx1 + gx0) / 2, (gy1 + gy0) / 2), tol=3
                ):
                    continue
                kdx_gg45.append((gx0, gy0, gx1, gy1))

        if 6 >= len(kdx_gg90mid) and 6 >= len(
            kdx_gg45
        ):  # 最后保证这种gangang肯定被return
            gangangs90[(kx0, ky0, kx1, ky1)] = (tuple(kdx_gg90mid), ())
            gangangs45[(kx0, ky0, kx1, ky1)] = ((), tuple(kdx_gg45))
            if len(kdx_gg90mid) >= 2:
                good_gg90_c += 1
            elif len(kdx_gg45) >= 2:
                good_gg45_c += 1

        else:
            print("-", end="")

    if good_gg90_c >= good_gg45_c:
        return gangangs90
    else:
        return gangangs45


# ------------------------task-3-------------------------


def get_exlines_from_txt(restkdx_path, exggs_path, exkdx_path):
    basekdx = []
    not_kdx = []
    kdx_lines = []

    with open(restkdx_path, "r", encoding="utf-8") as txt:
        for line in txt:
            basekdx.append(tuple(map(float, line.strip()[1:-1].split(","))))

    with open(exggs_path, "r", encoding="utf-8") as txt:
        for line in txt:
            not_kdx.append(tuple(map(float, line.strip()[1:-1].split(","))))

    with open(exkdx_path, "r", encoding="utf-8") as txt:
        for line in txt:
            kdx_lines.append(tuple(map(float, line.strip()[1:-1].split(","))))

    return basekdx, not_kdx, kdx_lines

'''
def _prune_cycles(adj):
    """
    去环策略：只移除造成环的无向边（non-tree edges），保留所有节点。
    做法：
      1) 邻接表规范化：去重/去自环/修复对称性（robust，避免假环）。
      2) 对每个连通分量做 DFS 建生成树（spanning tree）。
      3) 遇到已访问且非父边的 (u,v) 记为“非树边”，最后从两侧邻接表移除。
    原地修改 adj 并返回。
    """
    m = len(adj)

    # --- (1) 邻接表规范化（important，避免重复边/自环引起的误报） ---
    for u in range(m):
        # 去重、去自环
        adj[u] = sorted({v for v in adj[u] if v != u})
    # 修复无向对称
    for u in range(m):
        for v in adj[u]:
            if u not in adj[v]:
                adj[v].append(u)
    for u in range(m):
        adj[u].sort()

    visited  = [False] * m
    parent   = [-1]    * m
    to_remove = set()   # 记录需要移除的无向边(小,大)

    # --- (2) DFS 构造生成树，并标记“非树边” ---
    for s in range(m):
        if visited[s]:
            continue
        stack = [s]
        visited[s] = True
        parent[s] = -1

        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    stack.append(v)
                else:
                    # 已访问且不是父边 => 非树边（形成环）
                    a, b = (u, v) if u < v else (v, u)
                    to_remove.add((a, b))

    # --- (3) 从两侧邻接表移除这些造成环的边 ---
    if to_remove:
        for (u, v) in to_remove:
            if v in adj[u]:
                adj[u].remove(v)
            if u in adj[v]:
                adj[v].remove(u)

    return adj
'''

def _prune_cycles(adj):
    """
    一票否决版：只要图中存在至少一个环，就把该图中所有节点的邻接清空（等价于整块剔除）。
    用在 _connected_components 提供的 CC 子图上。
    """
    m = len(adj)

    # 可选的轻量规范化，避免重复边/自环引起的误判（不改也能用）
    for u in range(m):
        if adj[u]:
            adj[u] = list({v for v in adj[u] if v != u})

    visited = [False] * m

    def dfs(u: int, p: int) -> bool:
        visited[u] = True
        for v in adj[u]:
            if v == p:
                continue
            if not visited[v]:
                if dfs(v, u):
                    return True
            else:
                # 在无向图中遇到已访问且不是父亲 => 有环
                return True
        return False

    # 检测是否存在任意环
    has_cycle = False
    for s in range(m):
        if not visited[s] and adj[s]:
            if dfs(s, -1):
                has_cycle = True
                break

    # 一旦发现环：整块清空（“全部去掉”）
    if has_cycle:
        for u in range(m):
            adj[u].clear()

    return adj

# eps_pos决定端点是否相连/eps_ang决定线段方向是否一致/eps_line决定重合精度/eps_olap决定线段最小重叠限制
def _build_graph_kdx(kdx_lines, eps_pos=1, eps_ang=1, eps_line=1, eps_olap=1):
    _pt = lambda seg, i: (seg[0], seg[1]) if i == 0 else (seg[2], seg[3])
    _dist = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])

    def _colinear_overlap(s1, s2, eps_ang, eps_line, eps_olap):
        def _proj_interval(seg, axis):
            p0 = (seg[0], seg[1])
            p1 = (seg[2], seg[3])
            s0 = p0[0] * axis[0] + p0[1] * axis[1]
            s1 = p1[0] * axis[0] + p1[1] * axis[1]
            return (s0, s1) if s0 <= s1 else (s1, s0)

        def _dist_point_to_line(pt, seg):
            x0, y0, x1, y1 = seg
            px, py = pt
            vx, vy = x1 - x0, y1 - y0
            wx, wy = px - x0, py - y0
            v2 = vx * vx + vy * vy
            if v2 == 0.0:
                return math.hypot(px - x0, py - y0)
            return abs(wx * vy - wy * vx) / math.sqrt(v2)

        def _ang(seg):
            x0, y0, x1, y1 = seg
            return math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0

        a1 = _ang(s1)
        a2 = _ang(s2)
        d = abs(a1 - a2) % 180.0
        if (180.0 - d if d > 90.0 else d) > eps_ang:
            return False
        if _dist_point_to_line((s1[0], s1[1]), s2) > eps_line:
            return False
        if _dist_point_to_line((s1[2], s1[3]), s2) > eps_line:
            return False
        if _dist_point_to_line((s2[0], s2[1]), s1) > eps_line:
            return False
        if _dist_point_to_line((s2[2], s2[3]), s1) > eps_line:
            return False
        ax = (
            math.cos(math.radians((a1 + a2) / 2.0)),
            math.sin(math.radians((a1 + a2) / 2.0)),
        )
        s0, s1v = _proj_interval(s1, ax)
        t0, t1v = _proj_interval(s2, ax)
        return (min(s1v, t1v) - max(s0, t0)) >= eps_olap

    n = len(kdx_lines)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s1, s2 = kdx_lines[i], kdx_lines[j]
            did_endpoint_touch = any(
                _dist(_pt(s1, i), _pt(s2, j)) <= eps_pos for i in (0, 1) for j in (0, 1)
            )
            if did_endpoint_touch or _colinear_overlap(
                s1, s2, eps_ang, eps_line, eps_olap
            ):
                adj[i].append(j)
                adj[j].append(i)
    return adj


def _connected_components(adj, kdx_lines, basekdx):
    """
    先对每个原始 CC 做“去闭环”，再在去环后的子图上按 CC 拆分并检查 basekdx。
    返回每个 CC 对应的线段列表（去环后，且含 base，且大小>1）。
    """
    n = len(adj)
    vis = [False] * n
    comps = []
    base_set = set(map(tuple, basekdx))

    for i in range(n):
        if vis[i]:
            continue

        # ① 在原图上取出一个 CC 的所有“原索引”
        q = deque([i])
        vis[i] = True
        comp_nodes = []
        while q:
            u = q.popleft()
            comp_nodes.append(u)
            for v in adj[u]:
                if not vis[v]:
                    vis[v] = True
                    q.append(v)
        if len(comp_nodes) <= 1:
            continue

        # ② 构建该 CC 的“诱导子图”（重新编号 0..m-1）
        idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(comp_nodes)}
        m = len(comp_nodes)
        sub_adj = [[] for _ in range(m)]
        for orig_u in comp_nodes:
            u = idx_map[orig_u]
            for orig_v in adj[orig_u]:
                if orig_v in idx_map:
                    v = idx_map[orig_v]
                    sub_adj[u].append(v)

        # ③ 在该子图上“去闭环”（剥掉环上的节点）
        _prune_cycles(sub_adj)

        # ④ 在“去环后的子图”上再求 CC，并检查 basekdx
        sub_vis = [False] * m
        for s in range(m):
            if (sub_vis[s] or not sub_adj[s]):  # 被清空或孤点将被处理，但我们最后需要 size>1
                continue
            q = deque([s])
            sub_vis[s] = True
            sub_comp = []
            has_base = False

            while q:
                u = q.popleft()
                sub_comp.append(u)

                # 映射回原索引拿线段，再检查是否在 basekdx
                orig_u = comp_nodes[u]
                if tuple(kdx_lines[orig_u]) in base_set:
                    has_base = True

                for v in sub_adj[u]:
                    if not sub_vis[v]:
                        sub_vis[v] = True
                        q.append(v)

            # ⑤ 只收“去环后 size>1 且含 base”的子分量
            if len(sub_comp) > 1 and has_base:
                comps.append([kdx_lines[comp_nodes[u]] for u in sub_comp])

    return comps


def find_kdx_by_cc(basekdx, kdx_lines, gangang_lines, eps=(1, 1, 1, 1)):
    eps_pos, eps_ang, eps_line, eps_olap = eps
    L2_MIN = 25
    kdx_lines = [
        seg
        for seg in kdx_lines
        if (seg[2] - seg[0]) * (seg[2] - seg[0]) + (seg[3] - seg[1]) * (seg[3] - seg[1])
        >= L2_MIN**2
    ]

    adj = _build_graph_kdx(kdx_lines, eps_pos, eps_ang, eps_line, eps_olap)
    comps = _connected_components(adj, kdx_lines, basekdx)

    results = {}
    for comp in comps:
        gangangs_dict = get_gangangs(gangang_lines, comp, tol=2.2)
        gangangs = set()
        for gg90, gg45 in gangangs_dict.values():
            if len(gg90) > 1 or len(gg45) > 1:
                continue
            for g in gg90:
                gangangs.add(g)
            for g in gg45:
                gangangs.add(g)

        if len(gangangs) >= 2:
            results[tuple(comp)] = tuple(gangangs)
    return results
