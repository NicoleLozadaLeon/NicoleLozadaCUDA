#!/usr/bin/env python3
"""Click on the map to trace the street network. Saves nodes/streets/restaurants as JSON."""
import csv, json, math, os, unicodedata
import cv2

W, S, E, N = -63.20374411309951, -17.779011638450203, -63.18915387798629, -17.752162388363526
IMAGE = "data/equipetrol.jpg"
CSV_FILE = "data/restaurants_osm.csv"
OUT = "tools/network_draft.json"
ZOOM = 0.7          
VIEW_H = 900        
SCROLL_STEP = 120   
PICK_RADIUS = 12
CHOSEN = ["tgi fridays", "pizza hut", "pollo campeon", "sushi bar by slatkis", "tarbush", "la gaditana"]

def ascii_lower(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()

def merc_y(lat):
    return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

def latlon_to_px(lat, lon, w, h):
    px = (lon - W) / (E - W) * w
    py = (merc_y(N) - merc_y(lat)) / (merc_y(N) - merc_y(S)) * h
    return px, py

def px_to_latlon(px, py, w, h): 
    lon = W + px / w * (E - W)
    my = merc_y(N) - py /h * (merc_y(N)  - merc_y(S))
    lat = math.degrees(2 * math.atan(math.exp(my)) - math.pi / 2)
    return lat, lon 

def haversine_m(lat1, lon1, lat2, lon2): 
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2)** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(a))

class Net:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.restaurants = []
        self.selected = None
        self.oneway = False
        self.actions = []

    def nearest(self, px, py, radius):
        best, best_d = None, radius
        for i, (nx, ny) in enumerate(self.nodes):
            d = math.hypot(nx - px, ny - py)
            if d <= best_d:
                best, best_d = i, d
        return best

    def click(self, px, py, radius):
        old = self.selected
        idx = self.nearest(px, py, radius)
        if idx is None:
            self.nodes.append([px, py])
            self.actions.append("node")
            idx = len(self.nodes) - 1
        if old is not None and old != idx:
            exists = any((a == old and b == idx) or (a == idx and b == old) for a, b, _ in self.edges)
            if not exists:
                self.edges.append([old, idx, self.oneway])
                self.actions.append("edge")
        self.selected = idx    
    def undo(self): ...
    def save(self, w, h): ...
    def load(self, w, h): ...
    def assign_restaurants(self, candidates, w, h): ...

def load_candidates(): ...

def draw(view, net, scale, candidates, show_candidates):
    img = view.copy()
    def pt(i):
        return int(net.nodes[i][0] * scale), int(net.nodes[i][1] * scale)
    for a, b, oneway in net.edges:
        if oneway:
            cv2.arrowedLine(img, pt(a), pt(b), (0, 0, 255), 2, tipLength=0.15)
        else:
            cv2.line(img, pt(a), pt(b), (255, 0, 0), 2)
    for i in range(len(net.nodes)):
        color = (0, 255, 255) if i == net.selected else (0, 160, 0)
        x, y = pt(i)
        cv2.circle(img, (x, y), 4, color, -1)
        cv2.putText(img, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    n_oneway = sum(1 for e in net.edges if e[2])
    mode = "ONEWAY" if net.oneway else "two-way"
    text = f"nodes {len(net.nodes)}  edges {len(net.edges)}  oneway {n_oneway}  mode {mode}"
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return img

def main():
    img = cv2.imread(IMAGE)
    if img is None:
        print("cannot read", IMAGE)
        return
    h, w = img.shape[:2]
    scale = ZOOM
    view = cv2.resize(img, None, fx=scale, fy=scale)
    view_h = min(VIEW_H, view.shape[0])
    max_oy = view.shape[0] - view_h
    oy = 0
    net = Net()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            net.click(x / scale, (y + oy) / scale, PICK_RADIUS / scale)
        elif event == cv2.EVENT_RBUTTONDOWN:
            net.selected = None

    cv2.namedWindow("network")
    cv2.setMouseCallback("network", on_mouse)
    show_candidates = False
    while True:
        full = draw(view, net, scale, [], show_candidates)
        frame = full[oy:oy + view_h].copy()
        n_oneway = sum(1 for e in net.edges if e[2])
        status = f"nodes {len(net.nodes)}  edges {len(net.edges)}  oneway {n_oneway}  mode {'ONEWAY' if net.oneway else 'two-way'}  y {oy}/{max_oy}"
        cv2.putText(frame, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.imshow("network", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("o"):
            net.oneway = not net.oneway
        elif key == ord("n"):
            net.selected = None
        elif key == ord("k"):
            oy = min(max_oy, oy + SCROLL_STEP)
        elif key == ord("i"):
            oy = max(0, oy - SCROLL_STEP)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
