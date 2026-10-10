#!/usr/bin/env python3
"""Click on the map to trace the street network. Saves nodes/streets/restaurants as JSON."""
import csv, json, math, os, unicodedata
import cv2

W, S, E, N = -63.205248240852164, -17.77504707884483, -63.187351138452286, -17.750567044823004
IMAGE = "data/equipetrol.png"
CSV_FILE = "data/restaurants_osm.csv"
OUT = "tools/network_draft.json"
ZOOM = 1.0          
VIEW_H = 950        
VIEW_W = 1250         # width of the visible window
SCROLL_STEP = 120   
PICK_RADIUS = 12
CHOSEN = ["tradiciones", "pizza hut", "ambika", "menta", "yogen fruz", "sushi bar by slatkis"]
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

    def delete_selected(self):
        i = self.selected
        if i is None:
            return
        self.nodes.pop(i)
        self.edges = [[a - (a > i), b - (b > i), ow]
                      for a, b, ow in self.edges if a != i and b != i]
        self.restaurants = [dict(r, node=r["node"] - (r["node"] > i))
                            for r in self.restaurants if r["node"] != i]
        self.actions = []
        self.selected = None
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

    def undo(self):
        if not self.actions:
            return
        last = self.actions.pop()
        if last == "edge":
            self.edges.pop()
        else:
            self.nodes.pop()
        self.selected = None

    def save(self, w, h):
        nodes = []
        for i, (px, py) in enumerate(self.nodes):
            lat, lon = px_to_latlon(px, py, w, h)
            nodes.append({"id": f"n{i}", "lat": round(lat, 7), "lon": round(lon, 7)})
        streets = [{"id": f"s{i}", "from": f"n{a}", "to": f"n{b}", "oneWay": bool(ow)}
                   for i, (a, b, ow) in enumerate(self.edges)]
        restaurants = [{"id": f"r{i}", "name": r["name"], "node": f"n{r['node']}"}
                       for i, r in enumerate(self.restaurants)]
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump({"nodes": nodes, "streets": streets, "restaurants": restaurants},
                      f, indent=2, ensure_ascii=False)
        print(f"saved {len(nodes)} nodes, {len(streets)} streets, {len(restaurants)} restaurants to {OUT}")

    def load(self, w, h):
        if not os.path.exists(OUT):
            return
        with open(OUT, encoding="utf-8") as f:
            data = json.load(f)
        index = {}
        for i, n in enumerate(data["nodes"]):
            self.nodes.append(list(latlon_to_px(n["lat"], n["lon"], w, h)))
            index[n["id"]] = i
        for s in data["streets"]:
            self.edges.append([index[s["from"]], index[s["to"]], s["oneWay"]])
        for r in data.get("restaurants", []):
            self.restaurants.append({"name": r["name"], "node": index[r["node"]], "meters": 0.0})
        print(f"loaded {len(self.nodes)} nodes, {len(self.edges)} streets from {OUT}")

    def assign_restaurants(self, candidates, w, h):
        self.restaurants = []
        if not self.nodes:
            print("trace some nodes first")
            return
        for chosen in CHOSEN:
            found = [c for c in candidates if ascii_lower(c[2]) == chosen]
            if not found:
                print(f"{chosen}: not found in the CSV")
                continue
            lat, lon, name = found[0]
            best, best_m = None, float("inf")
            for i, (px, py) in enumerate(self.nodes):
                nlat, nlon = px_to_latlon(px, py, w, h)
                m = haversine_m(lat, lon, nlat, nlon)
                if m < best_m:
                    best, best_m = i, m
            warn = "   <-- WARNING: farther than 100 m" if best_m > 100 else ""
            print(f"{name} -> n{best}, {best_m:.0f} m{warn}")
            clean = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
            self.restaurants.append({"name": clean, "node": best, "meters": best_m})

def load_candidates():
    out = []
    with open(CSV_FILE, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)                       # skip the header line
        for row in reader:
            if len(row) >= 3:
                out.append((float(row[0]), float(row[1]), row[2]))
    return out

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
    if show_candidates:
        h_img, w_img = view.shape[:2]
        for lat, lon, name in candidates:
            px, py = latlon_to_px(lat, lon, w_img / scale, h_img / scale)
            x, y = int(px * scale), int(py * scale)
            chosen = ascii_lower(name) in CHOSEN
            if show_candidates == 1 and not chosen:
                continue
            color = (0, 140, 255) if chosen else (255, 0, 255)
            size = 6 if chosen else 4
            cv2.rectangle(img, (x - size, y - size), (x + size, y + size), color, -1)
            cv2.putText(img, ascii_lower(name)[:18], (x + 8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return img

def main():
    img = cv2.imread(IMAGE)
    if img is None:
        print("cannot read", IMAGE)
        return
    h, w = img.shape[:2]
    scale = ZOOM
    view = cv2.resize(img, None, fx=scale, fy=scale)
    faded = cv2.convertScaleAbs(view, alpha=0.6, beta=80)
    use_faded = True
    view_h = min(VIEW_H, view.shape[0])
    view_w = min(VIEW_W, view.shape[1])
    max_ox = view.shape[1] - view_w
    ox = 0
    max_oy = view.shape[0] - view_h
    oy = 0
    net = Net()
    net.load(w, h)
   
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            net.click((x + ox) / scale, (y + oy) / scale, PICK_RADIUS / scale)
        elif event == cv2.EVENT_RBUTTONDOWN:
            net.selected = None

    cv2.namedWindow("network")
    cv2.setMouseCallback("network", on_mouse)
    show_candidates = False
    candidates = load_candidates()
    while True:
        full = draw(faded if use_faded else view, net, scale, candidates, show_candidates)

        frame = full[oy:oy + view_h, ox:ox + view_w].copy()
        n_oneway = sum(1 for e in net.edges if e[2])
        status = f"nodes {len(net.nodes)}  edges {len(net.edges)}  oneway {n_oneway}  mode {'ONEWAY' if net.oneway else 'two-way'}  x {ox}/{max_ox}  y {oy}/{max_oy}"
        cv2.putText(frame, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.imshow("network", frame)
        key = cv2.waitKey(30) & 0xFF
       
        if key == ord("q"):
            net.save(w, h)
            break
        elif key == ord("f"):
            use_faded = not use_faded
        elif key == ord("u"):
            net.undo()
        elif key == ord("s"):
            net.save(w, h)
        elif key == ord("o"):
            net.oneway = not net.oneway
        elif key == ord("n"):
            net.selected = None
        elif key == ord("k"):
            oy = min(max_oy, oy + SCROLL_STEP)
        elif key == ord("i"):
            oy = max(0, oy - SCROLL_STEP)
        elif key == ord("j"):
            ox = max(0, ox - SCROLL_STEP)
        elif key == ord("l"):
            ox = min(max_ox, ox + SCROLL_STEP)
        elif key == ord("x"):
            net.delete_selected()
        elif key == ord("d"):
            show_candidates = (show_candidates + 1) % 3
        elif key == ord("r"):
            net.assign_restaurants(candidates, w, h)

    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
