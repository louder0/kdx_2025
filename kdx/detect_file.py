import math
from collections import deque, defaultdict

from coords_file import is_point_on_line


#------------------------task-0------------------------

def predict_qq_with_yolo(cut_imgs, model, qq_txt_path):
    circles = []
    for pngxy, img in cut_imgs:
        results = model(img)
        pngx, pngy = pngxy
        for result in results[0].boxes:
            if result.conf.cpu().tolist()[0]<0.7:
                continue
            x0,y0,x1,y1 = result.xyxy.cpu().tolist()[0]
            x0+=pngx
            x1+=pngx
            y0+=pngy
            y1+=pngy
            cx, cy = (x0+x1)/2,(y0+y1)/2
            circles.append((cx,cy))

    with open(qq_txt_path,'w',encoding='utf-8') as txt:
        for circle in circles:
            txt.write(f'{circle}\n')

    return circles



#------------------------task-1------------------------

def is_wanted_line(draw, features=['items', 'type', 'rect', 'color', 'width', 'dashes']):
    for feature in features:
        if feature not in draw:
            print(f'dict denied: {feature} not in dict')
            return False

    #items在get_drawings()里面测 把所有'l'揉到一起
    if draw['type'] != 's' and draw['type'] != 'fs':
        #print(f'dict denied: type is {draw["type"]}')
        return False    #绘图类型 's'代表描边
    #if draw['color'] != (0.0, 0.0, 0.0):
        #print(f'dict denied: color is {draw["color"]}')
        return False    #描边颜色 rgb 0-1
    if draw['stroke_opacity'] != 1.0:    #给qq6服务
        #print(f'dict denied: stroke_opacity is {draw["stroke_opacity"]}')
        return False    #描边透明度 0.0完全透明 - 1.0完全不透明
    if 'Hatch' in draw.get('layer', ''):
        return False

    
    rect = draw['rect']
    if abs(rect.x0-rect.x1)+abs(rect.y0-rect.y1) < 2:
        #print('dict denied: too small')
        return False

    return True
    '''
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
    '''



#------------------------task-2-------------------------

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
    dot = vx*wx + vy*wy
    # 防止浮点误差
    cosθ = max(-1.0, min(1.0, dot/(nv*nw)))
    θ = math.degrees(math.acos(cosθ))
    return [θ, 180.0 - θ]

def did_intersect(p1, p2, p3, p4, tol=2):
    x1,y1 = p1; x2,y2 = p2
    x3,y3 = p3; x4,y4 = p4

    # 叉积符号函数
    def ori(ax,ay, bx,by, cx,cy):
        return (bx-ax)*(cy-ay) - (by-ay)*(cx-ax)

    # 先做正常相交检测（X形、T形、端点接触）
    o1 = ori(x1,y1, x2,y2, x3,y3)
    o2 = ori(x1,y1, x2,y2, x4,y4)
    o3 = ori(x3,y3, x4,y4, x1,y1)
    o4 = ori(x3,y3, x4,y4, x2,y2)
    if o1*o2 <= 0 and o3*o4 <= 0:
        return True

    # 否则再检测“近似相交”：四个端点到对方线段的最小距离
    def point_to_seg_dist(px, py, x0, y0, x1, y1):
        vx, vy = x1-x0, y1-y0
        wx, wy = px-x0, py-y0
        # 投影系数
        t = (vx*wx + vy*wy) / (vx*vx + vy*vy) if (vx*vx+vy*vy)>0 else 0
        t = max(0.0, min(1.0, t))
        cx, cy = x0 + t*vx, y0 + t*vy
        return math.hypot(px-cx, py-cy)

    # 端点到线段的最小距离
    dists = [
        point_to_seg_dist(x1,y1, x3,y3,x4,y4),
        point_to_seg_dist(x2,y2, x3,y3,x4,y4),
        point_to_seg_dist(x3,y3, x1,y1,x2,y2),
        point_to_seg_dist(x4,y4, x1,y1,x2,y2),
    ]
    return min(dists) <= tol

def get_gangangs(not_kdx, kdx_lines, lens=((33,25,25), 55), tol=1):
    gangangs90 = {}
    gangangs45 = {}    #两个都可以是的 避免后面if/elif把这类扔掉
    good_gg90_c = 0
    good_gg45_c = 0

    min_len, max_len = lens

    for kx0,ky0,kx1,ky1 in kdx_lines:
        kdx_gg90mid = []
        kdx_gg45 = []
        for gx0,gy0,gx1,gy1 in not_kdx:
            p1,p2,p3,p4 = (kx0,ky0),(kx1,ky1),(gx0,gy0),(gx1,gy1)
            if not did_intersect(p1,p2,p3,p4):
                continue

            a1,a2 = get_cross_line_angles(p1,p2,p3,p4)
            is90d = abs(a1-90)<tol and abs(a2-90)<tol
            is45d = abs(min(a1,a2)-45)<tol and abs(max(a1,a2)-135)<tol
            line_len = math.hypot(abs(gx1 - gx0), abs(gy1 - gy0))
            #min_len依次是kdx_gg90mid, kdx_gg45和kdx_gg90

            if is90d:
                if not max_len > line_len > min_len[2]:
                    continue
                if not is_point_on_line(p1,p2,((gx1+gx0)/2,(gy1+gy0)/2),tol=3):   #tol为3的时候qq7的fp好像是没有||||||||||||||||||||||||||||
                    continue
                if not line_len > min_len[0]:
                    continue
                kdx_gg90mid.append((gx0,gy0,gx1,gy1))

            elif is45d:
                if not max_len > line_len > min_len[1]:
                    continue
                kdx_gg45.append((gx0,gy0,gx1,gy1))

        if 6>=len(kdx_gg90mid) and 6>=len(kdx_gg45):    #最后保证这种gangang肯定被return
            gangangs90[(kx0,ky0,kx1,ky1)] = (tuple(kdx_gg90mid), ())
            gangangs45[(kx0,ky0,kx1,ky1)] = ((), tuple(kdx_gg45))
            if len(kdx_gg90mid)>=2:
                good_gg90_c+=1
            elif len(kdx_gg45)>=2:
                good_gg45_c+=1
        
        else:
            print('-',end='')

    if good_gg90_c>=good_gg45_c:
        return gangangs90
    else:
        return gangangs45



#------------------------task-3-------------------------

def get_rexgangangs(restkdx_path, exggs_path, exkdx_path):
    basekdx_dict = dict()
    not_kdx = []
    kdx_lines = []
    with open(restkdx_path,'r',encoding='utf-8') as txt:
        for line in txt:
            s = line.strip()
            k,v = s.split(':')
            key = tuple(map(float, k[1:-1].split(',')))
            if v == '((), ())':
                val = ((),())
            else:
                val = tuple([
                            tuple(map(float, i[1:-2].split(',') )) 
                            if not i == '' 
                            else () 
                            for i in v[2:-2].split('), (')
                            ])
            basekdx_dict[key]=val

    with open(exggs_path,'r',encoding='utf-8') as txt:
        for line in txt:
            not_kdx.append(tuple(map(float, line.strip()[1:-1].split(','))))

    with open(exkdx_path,'r',encoding='utf-8') as txt:
        for line in txt:
            kdx_lines.append(tuple(map(float, line.strip()[1:-1].split(','))))
    
    return basekdx_dict, not_kdx, kdx_lines

if True:
    def _build_graph_kdx(kdx_lines, eps_pos=3.0, eps_ang=5.0, eps_line=2.0, eps_olap=3.0):
        """
        加速策略：
        1) 端点重合：用网格哈希，只在同格/邻格比较
        2) 共线重叠：角度分桶 + 法向距离分桶，在桶内做投影区间扫描
        返回：邻接表 adj（与旧版一致）
        """
        n = len(kdx_lines)
        if n == 0:
            return []

        # ---------- 预处理 ----------
        # 端点、方向角（0..180）、包围盒
        p0 = [(x0, y0) for (x0, y0, _,  _) in kdx_lines]
        p1 = [(x1, y1) for ( _,  _, x1, y1) in kdx_lines]
        ang = []
        bbox = []
        for (x0,y0,x1,y1) in kdx_lines:
            a = math.degrees(math.atan2(y1-y0, x1-x0)) % 180.0
            ang.append(a)
            xmin, xmax = (x0, x1) if x0 <= x1 else (x1, x0)
            ymin, ymax = (y0, y1) if y0 <= y1 else (y1, y0)
            # 略微外扩，避免边界抖动
            bbox.append((xmin-1e-6, ymin-1e-6, xmax+1e-6, ymax+1e-6))

        adj_sets = [set() for _ in range(n)]
        add_edge = lambda i, j: (adj_sets[i].add(j), adj_sets[j].add(i)) if i != j else None

        # ---------- 1) 端点重合：网格哈希 ----------
        cell = max(eps_pos, 1.0)
        to_cell = lambda x, y: (int(math.floor(x / cell)), int(math.floor(y / cell)))

        grid = defaultdict(list)  # (cx,cy) -> [(idx, which_endpoint)]
        for i in range(n):
            cx0, cy0 = to_cell(*p0[i])
            cx1, cy1 = to_cell(*p1[i])
            grid[(cx0, cy0)].append((i, 0))
            grid[(cx1, cy1)].append((i, 1))

        neighbors9 = [(dx,dy) for dx in (-1,0,1) for dy in (-1,0,1)]

        for (cx, cy), items in grid.items():
            # 取 3x3 邻域
            bucket = []
            for dx, dy in neighbors9:
                bucket.extend(grid.get((cx+dx, cy+dy), []))
            # 只在桶内做距离检查
            for a in range(len(items)):
                ia, wa = items[a]
                pa = p0[ia] if wa == 0 else p1[ia]
                for ib, wb in bucket:
                    if ib <= ia:  # 去重
                        continue
                    pb = p0[ib] if wb == 0 else p1[ib]
                    if math.hypot(pa[0]-pb[0], pa[1]-pb[1]) <= eps_pos:
                        add_edge(ia, ib)

        # ---------- 2) 共线重叠：角度/法向分桶 + 区间扫描 ----------
        bin_w = max(eps_ang, 1e-6)             # 角度桶宽
        to_bin = lambda a: int((a + 0.5*bin_w) // bin_w)  # 近似 round(a/bin_w)
        # 让 180° 回归 0°
        norm_bin = lambda b: 0 if (b * bin_w) >= 180.0 - 1e-9 else b

        # 行：角度桶（含左右邻桶），列：法向距离桶
        stripes = defaultdict(list)  # key=(b_angle, b_c) -> list of (s0, s1, idx)

        for i in range(n):
            b = norm_bin(to_bin(ang[i]))
            # 为覆盖“跨桶但角度差<=eps_ang”，把每条线同时放入 b-1, b, b+1 三个桶
            for bb in (b-1, b, b+1):
                bnorm = norm_bin(bb)
                theta = bnorm * bin_w  # 桶中心角
                u = (math.cos(math.radians(theta)), math.sin(math.radians(theta)))    # 主方向
                nvec = (-u[1], u[0])  # 法向
                # 法向距离 c，用端点之一即可
                c = nvec[0]*p0[i][0] + nvec[1]*p0[i][1]
                cbin = int(round(c / max(eps_line, 1e-6)))

                # 投影区间（在轴 u 上）
                s00 = u[0]*p0[i][0] + u[1]*p0[i][1]
                s01 = u[0]*p1[i][0] + u[1]*p1[i][1]
                s0, s1 = (s00, s01) if s00 <= s01 else (s01, s00)

                stripes[(bnorm, cbin)].append((s0, s1, i, u, nvec))

        # 在每个条带内做“按起点排序”的区间扫描，找重叠对
        for key, items in stripes.items():
            if len(items) <= 1:
                continue
            # 先按 s0 排序
            items.sort(key=lambda t: t[0])
            active = []  # 存 (s1, idx)
            j0 = 0
            for s0, s1, i, u, nvec in items:
                # 清理已不可能重叠的
                k = 0
                while k < len(active) and active[k][0] < s0 - eps_olap:
                    k += 1
                if k > 0:
                    active = active[k:]

                # 与当前 active 内所有区间重叠的加边
                for s1_prev, j in active:
                    # AABB 粗筛（可省，但几乎不要成本）
                    xmin_i, ymin_i, xmax_i, ymax_i = bbox[i]
                    xmin_j, ymin_j, xmax_j, ymax_j = bbox[j]
                    if xmax_i < xmin_j or xmax_j < xmin_i or ymax_i < ymin_j or ymax_j < ymin_i:
                        continue
                    # 区间重叠阈值
                    if min(s1, s1_prev) - s0 >= eps_olap:
                        add_edge(i, j)

                # 把自己放进 active（保持按 s1 非降序插入）
                # 简单起见：append 再按 s1 排一下；数量通常不大
                active.append((s1, i))
                active.sort(key=lambda t: t[0])

        # ---------- 输出邻接表（与旧版一致） ----------
        adj = [list(neis) for neis in adj_sets]
        return adj

    def _connected_components(adj, kdx_lines):
        n = len(adj)
        vis = [False]*n
        comps = []
        for i in range(n):
            if vis[i]: 
                continue
            q = deque([i]); vis[i] = True
            comp = []
            while q:
                u = q.popleft()
                comp.append(u)
                for v in adj[u]:
                    if not vis[v]:
                        vis[v] = True
                        q.append(v)
            comps.append(comp)
        return [[kdx_lines[i] for i in comp] for comp in comps]

def find_kdx_by_cc_simple(kdx_lines, gangang_lines, eps_pos=3.0, eps_ang=5.0, eps_line=2.0, eps_olap=3.0):
    adj = _build_graph_kdx(kdx_lines, eps_pos, eps_ang, eps_line, eps_olap)
    comps = _connected_components(adj, kdx_lines)

    results = {}
    for comp in comps:
        gangangs_dict = get_gangangs(gangang_lines, comp)
        gangangs = set()
        for gg90, gg45 in gangangs_dict.values():
            for g in gg90: 
                gangangs.add(g)
            for g in gg45: 
                gangangs.add(g)

        if 2 <= len(gangangs) <= 6:
            results[tuple(comp)]=tuple(gangangs)
    return results


